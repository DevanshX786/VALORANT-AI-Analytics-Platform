from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import os
import itertools

from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer
from src.scoring.map_score import MapScoreEngine
from src.scoring.chemistry import ChemistryEngine
from src.scoring.role_balance import RoleBalanceEngine
from src.models.baseline_model import BaselineModel


class MatchPredictRequest(BaseModel):
    team_a: List[str] = Field(..., min_items=5, max_items=5)
    team_b: List[str] = Field(..., min_items=5, max_items=5)
    map_pool: List[str] = Field(..., min_items=1)
    format: str = Field('Bo3', pattern='^(Bo1|Bo3|Bo5)$')
    stage: str = Field('Group')


app = FastAPI(
    title="VALORANT AI Analytics Platform API",
    version="0.1"
)


# ------------------------------------------------------------------
# Global preloaded resources (startup)
# ------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

loader = VCTDataLoader(data_dir=DATA_DIR)
ov = loader.load_overviews()
kills = loader.load_kills_stats()
maps_scores = loader.load_maps_scores()
scores = loader.load_scores()

# Keep all 5-year data in a single cleaned frame
if 'Year' not in ov.columns:
    ov['Year'] = 'unknown'

cleaner = VCTCleaner()

# Merge with kills to ensure all metrics in player scorer
join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Agents', 'Year']
merged = pd.merge(ov, kills, on=join_cols, how='inner')
clean_df = cleaner.clean_overviews(merged)

scorer = PlayerScorer()
scored_df = scorer.compute_overall_score(clean_df)

# Precompute player summary for quick lookup
player_summary = scored_df.groupby('Player').agg(
    mech_mean=('Mechanical_Score', 'mean'),
    clutch_mean=('Clutch_Score', 'mean'),
    entry_mean=('Entry_Success', 'mean'),
    util_mean=('Utility_Score', 'mean'),
    eco_mean=('Economic_Score', 'mean'),
    consistency_mean=('Consistency_Score', 'mean')
).reset_index().set_index('Player')

# Build map scoring engine for per-player/per-map computations
map_engine = MapScoreEngine()
map_engine.build(clean_df, maps_scores)

# Build chemistry engine based on all matches
chem_engine = ChemistryEngine(clean_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
role_engine = RoleBalanceEngine()

# Historical agent mapping per player (most common agent used)
agent_lookup = (
    clean_df.groupby('Player')['Agents']
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
    .to_dict()
)

# Load or create model
model = BaselineModel()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'baseline_xgb.json')
if os.path.exists(model_path):
    model.load(model_path)
else:
    raise RuntimeError('Trained model file not found: ' + model_path)


def _get_player_stats(name: str) -> Dict[str, float]:
    if name in player_summary.index:
        row = player_summary.loc[name]
        return {
            'mech_mean': float(row['mech_mean']),
            'clutch_mean': float(row['clutch_mean']),
            'entry_mean': float(row['entry_mean']),
            'util_mean': float(row['util_mean']),
            'eco_mean': float(row['eco_mean']),
            'consistency_mean': float(row['consistency_mean']),
        }
    # fallback neutral values if player not found
    return {
        'mech_mean': 50.0,
        'clutch_mean': 5.0,
        'entry_mean': 40.0,
        'util_mean': 5.0,
        'eco_mean': 10.0,
        'consistency_mean': 5.0,
    }


def build_team_features(players: List[str], map_name: str, stage: str, format: str) -> Dict[str, float]:
    # avoid repeats; keep 5 players as provided
    players = list(dict.fromkeys(players))[:5]

    # aggregate per-player features
    per_player = [_get_player_stats(p) for p in players]
    mech_values = [p['mech_mean'] for p in per_player]

    team_features = {
        'mech_mean': float(np.mean(mech_values)),
        'mech_max': float(np.max(mech_values)),
        'mech_std': float(np.std(mech_values, ddof=0)),
        'clutch_sum': float(np.sum([p['clutch_mean'] for p in per_player])),
        'entry_mean': float(np.mean([p['entry_mean'] for p in per_player])),
        'util_mean': float(np.mean([p['util_mean'] for p in per_player])),
        'eco_mean': float(np.mean([p['eco_mean'] for p in per_player])),
        'consistency': float(np.mean([p['consistency_mean'] for p in per_player])),
        'map_score': float(map_engine.get_team_map_score(players, map_name)),
        'chemistry': float(chem_engine.get_team_chemistry_score(players)),
        'role_balance': float(role_engine.compute_team_role_balance([agent_lookup.get(p, 'Unknown') for p in players])),
    }

    return team_features


def player_matchup_analysis(team_a: List[str], team_b: List[str], map_name: str) -> Dict:
    analysis = []
    for pa, pb in zip(team_a, team_b):
        score_a = map_engine.get_player_map_score(pa, map_name)['map_score']
        score_b = map_engine.get_player_map_score(pb, map_name)['map_score']
        analysis.append({
            'player_a': pa,
            'player_b': pb,
            'map_score_a': score_a,
            'map_score_b': score_b,
            'advantage': 'A' if score_a >= score_b else 'B'
        })
    return {'map': map_name, 'player_matchups': analysis}


@app.post('/predict/match')
def predict_match(req: MatchPredictRequest):
    team_a = req.team_a
    team_b = req.team_b
    map_pool = req.map_pool

    # Global preflight
    if len(team_a) != 5 or len(team_b) != 5:
        raise HTTPException(status_code=400, detail='Each team must have exactly 5 player names.')
    if set(team_a) & set(team_b):
        raise HTTPException(status_code=400, detail='Teams should not share players.')

    # Format map pool validation: maximum maps for the series
    fmt = req.format
    if fmt == 'Bo1' and len(map_pool) != 1:
        raise HTTPException(status_code=400, detail='Bo1 requires exactly 1 map.')
    if fmt == 'Bo3' and len(map_pool) > 3:
        raise HTTPException(status_code=400, detail='Bo3 allows at most 3 maps.')
    if fmt == 'Bo5' and len(map_pool) > 5:
        raise HTTPException(status_code=400, detail='Bo5 allows at most 5 maps.')
    if fmt == 'Bo3' and len(map_pool) < 1:
        raise HTTPException(status_code=400, detail='Bo3 requires at least 1 map.')
    if fmt == 'Bo5' and len(map_pool) < 1:
        raise HTTPException(status_code=400, detail='Bo5 requires at least 1 map.')

    map_results = []
    for map_name in map_pool:
        fa = build_team_features(team_a, map_name, req.stage, req.format)
        fb = build_team_features(team_b, map_name, req.stage, req.format)
        pred = model.predict_match(fa, fb)

        map_results.append({
            'map': map_name,
            'team_a_win_prob': pred['team_a_win_prob'],
            'team_b_win_prob': pred['team_b_win_prob'],
            'predicted_winner': pred['predicted_winner'],
            'player_matchup': player_matchup_analysis(team_a, team_b, map_name)
        })

    # overall aggregator by averaging map odds
    team_a_avg = float(np.mean([r['team_a_win_prob'] for r in map_results]))
    team_b_avg = float(np.mean([r['team_b_win_prob'] for r in map_results]))

    return {
        'team_a': team_a,
        'team_b': team_b,
        'map_pool': map_pool,
        'format': req.format,
        'stage': req.stage,
        'team_a_average_win_prob': team_a_avg,
        'team_b_average_win_prob': team_b_avg,
        'map_results': map_results,
        'recommended_bans': map_pool[:2],  # simple placeholder, from lowest team_a performance maybe
    }


@app.get('/players/search')
def players_search(q: str = Query(..., min_length=1, max_length=50), limit: int = 10):
    lower_q = q.strip().lower()
    candidates = [p for p in player_summary.index if lower_q in p.lower()]
    return {'query': q, 'results': candidates[:limit]}


@app.get('/player/{name}')
def player_detail(name: str):
    if name not in player_summary.index:
        raise HTTPException(status_code=404, detail='Player not found')

    row = player_summary.loc[name]
    fix = row.to_dict()
    res = {
        'player': name,
        'mechanical_score': float(fix['mech_mean']),
        'clutch_score': float(fix['clutch_mean']),
        'entry_score': float(fix['entry_mean']),
        'utility_score': float(fix['util_mean']),
        'economic_score': float(fix['eco_mean']),
        'consistency_score': float(fix['consistency_mean']),
        'agent': agent_lookup.get(name, 'Unknown'),
        'map_scores': {
            m: map_engine.get_player_map_score(name, m)['map_score']
            for m in ['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze', 'Lotus']
        }
    }
    return res


# Module 1: current roster forecasting endpoint (hardcoded roster list for MVP)
CURRENT_ROSTERS = {
    'Sentinels': ['TenZ', 'shahzam', 'zombs', 'nitr0', 'dapr'],
    'Liquid': ['aspas', 't3xture', 'Karon', 'Lakia', 'Meteor'],
    '100 Thieves': ['derke', 'yay', 'envy', 'Mistic', 'SicK'],
    'FaZe': ['frozen', 'f0rsakeN', 'sacy', 'nAts', 'Derrek'],
}


@app.get('/predict/current-rosters')
def predict_current_rosters(format: str = Query('Bo3', regex='^(Bo1|Bo3|Bo5)$'), stage: str = 'Group', map_pool: Optional[List[str]] = Query(None)):
    if map_pool is None or len(map_pool) == 0:
        map_pool = ['ascent', 'bind', 'haven']

    if format == 'Bo1' and len(map_pool) != 1:
        raise HTTPException(status_code=400, detail='Bo1 requires exactly 1 map in map_pool.')
    if format == 'Bo3' and len(map_pool) > 3:
        raise HTTPException(status_code=400, detail='Bo3 allows at most 3 maps.')
    if format == 'Bo5' and len(map_pool) > 5:
        raise HTTPException(status_code=400, detail='Bo5 allows at most 5 maps.')

    teams = list(CURRENT_ROSTERS.keys())
    matchups = []

    for team_a, team_b in itertools.combinations(teams, 2):
        team_a_roster = CURRENT_ROSTERS[team_a]
        team_b_roster = CURRENT_ROSTERS[team_b]

        # existing endpoint logic reuse
        req = MatchPredictRequest(team_a=team_a_roster, team_b=team_b_roster, map_pool=map_pool, format=format, stage=stage)
        prediction = predict_match(req)

        matchups.append({
            'team_a': team_a,
            'team_b': team_b,
            'team_a_roster': team_a_roster,
            'team_b_roster': team_b_roster,
            'team_a_avg_win_prob': prediction['team_a_average_win_prob'],
            'team_b_avg_win_prob': prediction['team_b_average_win_prob'],
            'predicted_winner': 'team_a' if prediction['team_a_average_win_prob'] >= prediction['team_b_average_win_prob'] else 'team_b',
            'map_results': prediction['map_results']
        })

    return {
        'format': format,
        'stage': stage,
        'map_pool': map_pool,
        'teams': teams,
        'matchups': matchups
    }
