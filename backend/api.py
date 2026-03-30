from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
from src.models.prediction_engine import PredictionEngine


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

# Add CORS Middleware to ensure the upcoming React frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Global preloaded resources (startup)
# ------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Map normalization: accept common variants, canonicalize to dataset naming.
# IMPORTANT: canonical form should match the `Map` strings produced by `MapScoreEngine.build()`.
MAP_NAME_ALIASES: Dict[str, str] = {
    # common casing/whitespace handled by normalize_map_name()
    'ice box': 'Icebox',
}


def normalize_map_name(map_name: str) -> str:
    if map_name is None:
        return ''
    key = str(map_name).strip()
    if not key:
        return ''
    # normalize to "Title" but preserve internal spacing
    key_norm = " ".join(key.split()).lower()
    if key_norm in MAP_NAME_ALIASES:
        return MAP_NAME_ALIASES[key_norm]
    return " ".join([w.capitalize() for w in key_norm.split(" ")])


def apply_series_pressure_adjustment(
    prob_a: float,
    fmt: str,
    team_a_wins: int,
    team_b_wins: int
) -> tuple[float, float]:
    """
    Apply a small state-aware adjustment for multi-map series.
    Idea:
      - trailing team gets a small "urgency" boost
      - leading team gets a smaller "momentum/closing" boost
    Net effect is intentionally small so it doesn't overpower core skill features.
    Returns: (adjusted_prob_a, delta_applied_to_prob_a)
    """
    if fmt not in ('Bo3', 'Bo5'):
        return prob_a, 0.0

    wins_needed = 2 if fmt == 'Bo3' else 3
    score_diff = team_a_wins - team_b_wins
    if score_diff == 0:
        return prob_a, 0.0

    trailing_is_a = score_diff < 0
    leader_wins = max(team_a_wins, team_b_wins)
    is_elimination_map = leader_wins == (wins_needed - 1)

    # Urgency: trailing team tries harder (slightly stronger effect).
    urgency_base = 0.03 if fmt == 'Bo3' else 0.02
    urgency_elim_bonus = 0.01 if is_elimination_map else 0.0
    urgency = urgency_base + urgency_elim_bonus

    # Momentum: leading team tends to close better / has confidence (smaller offsetting effect).
    # This is strongest in Bo5 when up 2-0 (classic "sweep momentum" vs opponent desperation).
    momentum = 0.0
    if fmt == 'Bo5':
        if (team_a_wins, team_b_wins) in [(2, 0), (0, 2)]:
            momentum = 0.01
        elif (team_a_wins, team_b_wins) in [(2, 1), (1, 2)]:
            momentum = 0.005
    elif fmt == 'Bo3':
        if (team_a_wins, team_b_wins) in [(1, 0), (0, 1)]:
            momentum = 0.005

    # Convert to a delta on Team A probability.
    # - If Team A is trailing: +urgency for A, but -momentum (since B is leading)
    # - If Team A is leading: -urgency for B, but +momentum for A
    delta = (urgency - momentum) if trailing_is_a else -(urgency - momentum)
    adjusted = max(0.01, min(0.99, prob_a + delta))
    return adjusted, delta

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
base_join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Year']
join_cols = [c for c in base_join_cols if c in ov.columns and c in kills.columns]
if not join_cols:
    raise RuntimeError("No common join columns found between overview and kills_stats.")

# Use a left join to avoid silent row loss when kills_stats is missing for some matches.
merged = pd.merge(ov, kills, on=join_cols, how='left', suffixes=('', '_kills'))

# Ensure `Agents` column exists post-merge (prefer overview's Agents)
if 'Agents' not in merged.columns:
    # common case: overview had Agents, but got suffixes from merge
    if 'Agents_x' in merged.columns:
        merged['Agents'] = merged['Agents_x']
    elif 'Agents_y' in merged.columns:
        merged['Agents'] = merged['Agents_y']
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

# Supported map set based on what actually exists in the built table.
try:
    SUPPORTED_MAPS = sorted(
        list(map_engine._player_map_table.index.get_level_values('Map').unique())  # type: ignore[attr-defined]
    )
except Exception:
    SUPPORTED_MAPS = []

# Build chemistry engine based on all matches
chem_engine = ChemistryEngine(clean_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
role_engine = RoleBalanceEngine()
pred_engine = PredictionEngine()

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

# ------------------------------------------------------------------
# Load current roster table (tier1_rosters.csv) and mapping helpers
# ------------------------------------------------------------------
TIER1_ROSTERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'tier1_rosters.csv')


def _load_tier1_rosters() -> pd.DataFrame:
    if not os.path.exists(TIER1_ROSTERS_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(TIER1_ROSTERS_PATH, encoding='latin1', keep_default_na=False)
    except Exception:
        return pd.read_csv(TIER1_ROSTERS_PATH, encoding='utf-8', errors='ignore', keep_default_na=False)


tier1_rosters_df = _load_tier1_rosters()

# roster_lookup maps lowercase name/id -> canonical player_name
roster_lookup: Dict[str, str] = {}
for _, r in tier1_rosters_df.iterrows():
    player_name = str(r.get('player_name', '')).strip()
    if player_name:
        roster_lookup[player_name.lower()] = player_name
    # player_id(s)
    for id_col in ['player_id', 'player_id_mapped', 'mapped_ids']:
        if id_col in r and r[id_col] not in [None, '', float('nan')]:
            ids = str(r[id_col]).split(',')
            for id_val in ids:
                clean_id = str(id_val).strip()
                if clean_id:
                    roster_lookup[clean_id.lower()] = player_name

# map all existing participants to lowercase normalization
player_summary_lower = {p.lower(): p for p in player_summary.index}


def resolve_player_name(player_identifier: str) -> Optional[str]:
    if player_identifier is None:
        return None
    ident = str(player_identifier).strip()
    if not ident:
        return None

    # Dataset Imposter Override: intercept 'f0rsaken' so it skips the dummy Sova account in the index 
    # and properly locks onto the Jett/Flex PRX superstar 'f0rsakeN'
    if ident.lower() == 'f0rsaken':
        return 'f0rsakeN'

    # exact dictionary key
    if ident in player_summary.index:
        return ident

    key = ident.lower()
    if key in player_summary_lower:
        return player_summary_lower[key]
    if key in roster_lookup:
        return roster_lookup[key]

    # numeric conflict or extra characters
    if key.isdigit() and key in roster_lookup:
        return roster_lookup[key]

    return None


def get_current_roster_by_team(team_name: str) -> List[str]:
    if not team_name:
        return []

    # prefer CSV roster if available
    if not tier1_rosters_df.empty:
        team_tb = tier1_rosters_df[tier1_rosters_df['team_name'].astype(str).str.lower() == team_name.lower()]
        if not team_tb.empty:
            return [str(x).strip() for x in team_tb['player_name'] if str(x).strip()]

    # If tier1_rosters.csv has no matching team, return empty roster
    return []


def get_all_current_rosters() -> Dict[str, List[str]]:
    if not tier1_rosters_df.empty:
        grouped = tier1_rosters_df.groupby(tier1_rosters_df['team_name'].astype(str).str.strip().str.title())['player_name']
        return {team: [str(p).strip() for p in players if str(p).strip()] for team, players in grouped}

    # No static fallback roster; rely entirely on tier1_rosters.csv source
    return {}


def _get_player_stats(name: str) -> Dict[str, float]:
    resolved_name = resolve_player_name(name)
    if resolved_name and resolved_name in player_summary.index:
        row = player_summary.loc[resolved_name]
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

    # resolve by name/id mapping
    resolved_players = []
    for p in players:
        resolved = resolve_player_name(p)
        if resolved:
            resolved_players.append(resolved)
        else:
            resolved_players.append(str(p).strip())

    # aggregate per-player features
    per_player = [_get_player_stats(p) for p in resolved_players]
    mech_values = [p['mech_mean'] for p in per_player]

    canonical_map = normalize_map_name(map_name)
    if SUPPORTED_MAPS and canonical_map not in SUPPORTED_MAPS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown map '{map_name}'. Supported maps: {SUPPORTED_MAPS}"
        )

    team_features = {
        'mech_mean': float(np.mean(mech_values)),
        'mech_max': float(np.max(mech_values)),
        'mech_std': float(np.std(mech_values, ddof=0)),
        'clutch_sum': float(np.sum([p['clutch_mean'] for p in per_player])),
        'entry_mean': float(np.mean([p['entry_mean'] for p in per_player])),
        'util_mean': float(np.mean([p['util_mean'] for p in per_player])),
        'eco_mean': float(np.mean([p['eco_mean'] for p in per_player])),
        'consistency': float(np.mean([p['consistency_mean'] for p in per_player])),
        'map_score': float(map_engine.get_team_map_score(resolved_players, canonical_map)),
        'chemistry': float(chem_engine.get_team_chemistry_score(resolved_players)),
        'role_balance': float(role_engine.compute_team_role_balance([agent_lookup.get(p, 'Unknown') for p in resolved_players])),
    }

    return pred_engine.apply_modifiers(team_features, fmt=format, stage=stage)


def player_matchup_analysis(team_a: List[str], team_b: List[str], map_name: str) -> Dict:
    canonical_map = normalize_map_name(map_name)
    analysis = []
    for pa, pb in zip(team_a, team_b):
        resolved_pa = resolve_player_name(pa) or pa
        resolved_pb = resolve_player_name(pb) or pb
        score_a = map_engine.get_player_map_score(resolved_pa, canonical_map)['map_score']
        score_b = map_engine.get_player_map_score(resolved_pb, canonical_map)['map_score']
        if score_a == score_b:
            advantage = 'E'
        else:
            advantage = 'A' if score_a > score_b else 'B'
        analysis.append({
            'player_a': pa,
            'player_b': pb,
            'map_score_a': score_a,
            'map_score_b': score_b,
            'advantage': advantage
        })
    return {'map': canonical_map, 'player_matchups': analysis}


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
    team_a_series_wins = 0
    team_b_series_wins = 0
    wins_needed = 1 if req.format == 'Bo1' else 2 if req.format == 'Bo3' else 3

    for map_name_in in map_pool:
        pre_a_wins = team_a_series_wins
        pre_b_wins = team_b_series_wins
        map_name = normalize_map_name(map_name_in)
        if SUPPORTED_MAPS and map_name not in SUPPORTED_MAPS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown map '{map_name_in}'. Supported maps: {SUPPORTED_MAPS}"
            )
        fa = build_team_features(team_a, map_name, req.stage, req.format)
        fb = build_team_features(team_b, map_name, req.stage, req.format)
        pred = model.predict_match(fa, fb)
        base_prob_a = float(pred['team_a_win_prob'])
        adj_prob_a, pressure_delta = apply_series_pressure_adjustment(
            base_prob_a, req.format, pre_a_wins, pre_b_wins
        )
        adj_prob_b = 1.0 - adj_prob_a
        predicted_winner = 'Team A' if adj_prob_a >= 0.5 else 'Team B'

        # Advance predicted series state for subsequent map context.
        if predicted_winner == 'Team A':
            team_a_series_wins += 1
        else:
            team_b_series_wins += 1

        map_results.append({
            'map': map_name,
            'team_a_win_prob': round(adj_prob_a * 100, 3),
            'team_b_win_prob': round(adj_prob_b * 100, 3),
            'predicted_winner': predicted_winner,
            'series_state_before': f"{pre_a_wins}-{pre_b_wins}",
            'pressure_adjustment_pct_points_for_team_a': round(pressure_delta * 100, 2),
            'player_matchup': player_matchup_analysis(team_a, team_b, map_name)
        })

        # Stop once a team has clinched the series in Bo3/Bo5.
        if req.format in ('Bo3', 'Bo5') and (team_a_series_wins >= wins_needed or team_b_series_wins >= wins_needed):
            break

    # overall aggregator by averaging map odds
    team_a_avg = round(float(np.mean([r['team_a_win_prob'] for r in map_results])), 3)
    team_b_avg = round(float(np.mean([r['team_b_win_prob'] for r in map_results])), 3)

    return {
        'team_a': team_a,
        'team_b': team_b,
        'map_pool': map_pool,
        'format': req.format,
        'stage': req.stage,
        'team_a_average_win_prob': team_a_avg,
        'team_b_average_win_prob': team_b_avg,
        'map_results': map_results
    }


@app.get('/players/search')
def players_search(q: str = Query(..., min_length=1, max_length=50), limit: int = 10):
    lower_q = q.strip().lower()
    candidates = [p for p in player_summary.index if lower_q in p.lower()]
    return {'query': q, 'results': candidates[:limit]}


@app.get('/rosters')
def get_rosters():
    rosters = get_all_current_rosters()
    return {'teams': list(rosters.keys()), 'rosters': rosters}


@app.get('/player/{name}')
def player_detail(name: str):
    resolved = resolve_player_name(name)
    if not resolved or resolved not in player_summary.index:
        raise HTTPException(status_code=404, detail='Player not found')

    row = player_summary.loc[resolved]
    fix = row.to_dict()
    res = {
        'player': resolved,
        'mechanical_score': float(fix['mech_mean']),
        'clutch_score': float(fix['clutch_mean']),
        'entry_score': float(fix['entry_mean']),
        'utility_score': float(fix['util_mean']),
        'economic_score': float(fix['eco_mean']),
        'consistency_score': float(fix['consistency_mean']),
        'agent': agent_lookup.get(resolved, 'Unknown'),
        'map_scores': {
            m: map_engine.get_player_map_score(resolved, m)['map_score']
            for m in (SUPPORTED_MAPS if SUPPORTED_MAPS else ['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze'])
        }
    }
    return res


# Module 1: current roster forecasting endpoint (tier1_rosters.csv source + fallback hardcoded)


@app.get('/predict/team-vs-team')
def predict_team_vs_team(
    team_a: str = Query(..., min_length=1),
    team_b: str = Query(..., min_length=1),
    format: str = Query('Bo3', regex='^(Bo1|Bo3|Bo5)$'),
    stage: str = 'Group',
    map_pool: Optional[List[str]] = Query(None)
):
    if map_pool is None or len(map_pool) == 0:
        map_pool = ['Ascent', 'Bind', 'Haven']

    if format == 'Bo1' and len(map_pool) != 1:
        raise HTTPException(status_code=400, detail='Bo1 requires exactly 1 map in map_pool.')
    if format == 'Bo3' and len(map_pool) > 3:
        raise HTTPException(status_code=400, detail='Bo3 allows at most 3 maps.')
    if format == 'Bo5' and len(map_pool) > 5:
        raise HTTPException(status_code=400, detail='Bo5 allows at most 5 maps.')

    rosters = get_all_current_rosters()
    normalized = {k.lower(): v for k, v in rosters.items()}

    team_a_key = team_a.strip().lower()
    team_b_key = team_b.strip().lower()

    if team_a_key not in normalized:
        raise HTTPException(status_code=404, detail=f'Team A not found: {team_a}')
    if team_b_key not in normalized:
        raise HTTPException(status_code=404, detail=f'Team B not found: {team_b}')
    if team_a_key == team_b_key:
        raise HTTPException(status_code=400, detail='Team A and Team B must be different.')

    team_a_roster = normalized[team_a_key][:5]
    team_b_roster = normalized[team_b_key][:5]

    if len(team_a_roster) != 5 or len(team_b_roster) != 5:
        raise HTTPException(status_code=400, detail='Both teams must have exactly 5 players in roster.')

    req = MatchPredictRequest(
        team_a=team_a_roster,
        team_b=team_b_roster,
        map_pool=map_pool,
        format=format,
        stage=stage
    )
    prediction = predict_match(req)

    return {
        **prediction,
        'team_a': team_a,
        'team_b': team_b,
        'team_a_roster': team_a_roster,
        'team_b_roster': team_b_roster,
        'format': format,
        'stage': stage,
        'map_pool': map_pool
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

    teams = list(get_all_current_rosters().keys())
    rosters = get_all_current_rosters()
    matchups = []

    for team_a, team_b in itertools.combinations(teams, 2):
        team_a_roster = rosters.get(team_a, [])[:5]
        team_b_roster = rosters.get(team_b, [])[:5]

        if len(team_a_roster) != 5 or len(team_b_roster) != 5:
            continue

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
