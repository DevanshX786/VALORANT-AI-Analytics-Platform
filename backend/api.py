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


def apply_series_pressure_adjustment(prob_a: float, fmt: str, team_a_wins: int, team_b_wins: int) -> Tuple[float, float]:
    """
    Applies situational 'Pressure' and 'Momentum' based on the series state.
    Bo1: Disables all modifiers.
    Bo3: Trailers (down 0-1) get a high 4.0% Pressure boost.
    Bo5: Trailers get 2.5% (down 1) or 5.0% (down 2) Pressure. 
         Leaders (up 2+) get a 2.0% Momentum boost.
    """
    if fmt == 'Bo1':
        return prob_a, 0.0

    score_diff = team_a_wins - team_b_wins
    if score_diff == 0:
        return prob_a, 0.0

    trailing_is_a = score_diff < 0
    abs_diff = abs(score_diff)
    
    pressure = 0.0
    momentum = 0.0

    if fmt == 'Bo3':
        # Rule: One map down gets a high 4% pressure boost
        if abs_diff == 1:
            pressure = 0.04
            
    elif fmt == 'Bo5':
        # Rule: Pressure scales with desperation (2.5% -> 5.0%)
        if abs_diff == 1:
            pressure = 0.025
        elif abs_diff >= 2:
            pressure = 0.05
            
        # Rule: Leaders get momentum after being 2 maps up
        if abs_diff >= 2:
            momentum = 0.02

    # Final Delta Calculation for Team A:
    # - If A is trailing: +pressure for A, BUT subtract leader B's momentum
    # - If A is leading: +momentum for A, BUT subtract leader B's pressure
    if trailing_is_a:
        delta = pressure - momentum
    else:
        delta = momentum - pressure

    adjusted = max(0.01, min(0.99, prob_a + delta))
    return float(adjusted), float(delta)

# ------------------------------------------------------------------
# Global Resource Loading (Hybrid Production-Lite Flow)
# Priority 1: Load precomputed pickles (Production/Cloud)
# Priority 2: Fallback to raw 1GB CSV ingestion (Local Dev)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Global Resource Loading (Hybrid Ultra-Lite Flow for 512MB RAM)
# ------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ULTRA_LITE_PATH = os.path.join(MODELS_DIR, 'vct_production.all.pkl')

player_summary = {}      # dict lookup
map_lookup = {}          # dict lookup {(player, map): stats}
chem_history = {}        # dict lookup
agent_lookup = {}        # dict lookup
ELITE_MECH_THRESHOLD = 45.0
SUPPORTED_MAPS = ['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze', 'Lotus', 'Pearl', 'Fracture', 'Sunset', 'Abyss', 'Corrode']

if os.path.exists(ULTRA_LITE_PATH):
    import pickle
    print(f"[Sync Engine] Ultra-Lite Production Mode detected. Purging heavy Pandas requirements...")
    print(f"[Sync Engine] Ultra-Lite Mode detect. Purging heavy Pandas requirements...")
    with open(ULTRA_LITE_PATH, 'rb') as f:
        pkg = pickle.load(f)
        player_summary = pkg['player_summary']
        map_lookup = pkg['map_lookup']
        chem_history = pkg['chemistry_history']
        agent_lookup = pkg['agent_lookup']
        ELITE_MECH_THRESHOLD = pkg['metadata']['elite_quantile_95']
    print(f"[Sync Engine] Success. {len(player_summary)} players and {len(map_lookup)} map profiles ready in RAM.")
else:
    print(f"[Sync Engine] No Ultra-Lite package found. Running full 1.3GB ingestion (Local Dev Only)...")
    loader = VCTDataLoader(data_dir=DATA_DIR)
    ov = loader.load_overviews()
    kills = loader.load_kills_stats()
    maps_scores = loader.load_maps_scores()
    
    cleaner = VCTCleaner()
    base_join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Year']
    join_cols = [c for c in base_join_cols if c in ov.columns and c in kills.columns]
    
    if join_cols:
        merged = pd.merge(ov, kills, on=join_cols, how='left', suffixes=('', '_kills'))
        if 'Agents' not in merged.columns:
            if 'Agents_x' in merged.columns: merged['Agents'] = merged['Agents_x']
        
        clean_df = cleaner.clean_overviews(merged)
        scorer = PlayerScorer()
        scored_df = scorer.compute_overall_score(clean_df)
        
        player_summary = scored_df.groupby('Player').agg(
            mech_mean=('Mechanical_Score', 'mean'),
            clutch_mean=('Clutch_Score', 'mean'),
            entry_mean=('Entry_Success', 'mean'),
            util_mean=('Utility_Score', 'mean'),
            eco_mean=('Economic_Score', 'mean'),
            consistency_mean=('Consistency_Score', 'mean')
        ).to_dict('index')
        
        map_engine_tmp = MapScoreEngine()
        map_engine_tmp.build(clean_df, maps_scores)
        map_lookup = map_engine_tmp._player_map_table.to_dict('index')
        
        chem_engine_tmp = ChemistryEngine(clean_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
        chem_history = chem_engine_tmp._pair_history

        agent_lookup = (
            clean_df.groupby('Player')['Agents']
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
            .to_dict()
        )
        ELITE_MECH_THRESHOLD = float(scored_df.groupby('Player')['Mechanical_Score'].mean().quantile(0.95))

# --- Ultra-Lite Wrappers for Production Efficiency ---
class ProMapEngine:
    def get_player_map_score(self, player, m):
        return map_lookup.get((player, m), {'map_score': 10.0, 'map_acs_mean': 180.0, 'map_kd_mean': 0.9})
    def get_team_map_score(self, players, m):
        scores = [self.get_player_map_score(p, m)['map_score'] for p in players]
        return np.mean(scores) if scores else 10.0

class ProChemEngine:
    FLOOR = 0.60
    def get_roster_chemistry(self, players):
        if len(players) < 2: return {'chemistry_score': self.FLOOR}
        counts = []
        for pair in combinations(players, 2):
            counts.append(chem_history.get(frozenset(pair), 0))
        avg_m = np.mean(counts)
        score = 0.6 + (min(avg_m, 20) / 20.0) * 0.35
        return {'chemistry_score': float(score)}

map_engine = ProMapEngine()
chem_engine = ProChemEngine()
role_engine = RoleBalanceEngine()
pred_engine = PredictionEngine()

model = BaselineModel()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'baseline_xgb.json')
if os.path.exists(model_path):
    model.load(model_path)
else:
    raise RuntimeError('Trained model file not found: ' + model_path)

# ------------------------------------------------------------------
# Resource Resolution Helpers
# ------------------------------------------------------------------
TIER1_ROSTERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'tier1_rosters.csv')

def _load_tier1_rosters() -> pd.DataFrame:
    if not os.path.exists(TIER1_ROSTERS_PATH): return pd.DataFrame()
    try:
        return pd.read_csv(TIER1_ROSTERS_PATH, encoding='latin1', keep_default_na=False)
    except:
        return pd.read_csv(TIER1_ROSTERS_PATH, encoding='utf-8', errors='ignore', keep_default_na=False)

tier_rosters_df = _load_tier1_rosters()
roster_lookup: Dict[str, str] = {}
if not tier_rosters_df.empty:
    for _, r in tier_rosters_df.iterrows():
        p_name = str(r.get('player_name', '')).strip()
        if p_name: roster_lookup[p_name.lower()] = p_name
        for i_col in ['player_id', 'player_id_mapped', 'mapped_ids']:
            if i_col in r and r[i_col] not in [None, '', float('nan')]:
                for id_val in str(r[i_col]).split(','):
                    clean_id = str(id_val).strip()
                    if clean_id: roster_lookup[clean_id.lower()] = p_name

player_summary_lower = {p.lower(): p for p in player_summary.keys()}


def get_all_current_rosters() -> Dict[str, List[str]]:
    if not tier_rosters_df.empty:
        grouped = tier_rosters_df.groupby(tier_rosters_df['team_name'].astype(str).str.strip().str.title())['player_name']
        return {team: [str(p).strip() for p in players if str(p).strip()] for team, players in grouped}
    return {}


def resolve_player_name(player_identifier: str) -> Optional[str]:
    if not player_identifier: return None
    ident = str(player_identifier).strip().lower()
    if ident in roster_lookup: return roster_lookup[ident]
    if ident in player_summary_lower: return player_summary_lower[ident]
    return None

def _get_player_stats(name: str) -> Dict[str, float]:
    resolved_name = resolve_player_name(name)
    stats = player_summary.get(resolved_name or name, {
        'mech_mean': 12.0, 'clutch_mean': 2.0, 'entry_mean': 10.0,
        'util_mean': 3.0, 'eco_mean': 5.0, 'consistency_mean': 5.0,
    }).copy()

    is_t1 = name.lower() in roster_lookup or (resolved_name and resolved_name.lower() in roster_lookup)
    if is_t1:
        stats['mech_mean'] = max(stats['mech_mean'], 22.0) + 5.0
        stats['clutch_mean'] += 2.0
        stats['util_mean'] += 2.0

    agent = agent_lookup.get(resolved_name or name, 'Unknown')
    role = role_engine.assign_role(agent)
    
    role_offset = 0.0
    if role in ('controller', 'sentinel'):
        role_offset, stats['util_mean'] = 15.0, stats['util_mean'] + 6.0
    elif role == 'initiator':
        role_offset, stats['util_mean'] = 10.0, stats['util_mean'] + 4.0
    elif role == 'duelist':
        role_offset = 2.0
    
    stats['mech_mean'] += role_offset
    if stats['mech_mean'] >= ELITE_MECH_THRESHOLD:
        stats['mech_mean'] = max(stats['mech_mean'], 45.0)
    
    return stats

def build_team_features(players: List[str], map_name: str, stage: str, format: str) -> Dict[str, float]:
    players = list(dict.fromkeys(players))[:5]
    resolved_players = [resolve_player_name(p) or str(p).strip() for p in players]

    per_player = [_get_player_stats(p) for p in resolved_players]
    mech_values = [p['mech_mean'] for p in per_player]
    canonical_map = normalize_map_name(map_name)
    
    team_features = {
        'mech_mean': float(np.mean(mech_values)),
        'mech_max': float(np.max(mech_values)),
        'mech_std': float(np.std(mech_values)),
        'clutch_sum': float(np.sum([p['clutch_mean'] for p in per_player])),
        'entry_mean': float(np.mean([p['entry_mean'] for p in per_player])),
        'util_mean': float(np.mean([p['util_mean'] for p in per_player])),
        'eco_mean': float(np.mean([p['eco_mean'] for p in per_player])),
        'consistency': float(np.mean([p['consistency_mean'] for p in per_player])),
        'map_score': float(map_engine.get_team_map_score(resolved_players, canonical_map)),
        'chemistry': float(chem_engine.get_roster_chemistry(resolved_players)['chemistry_score']),
        'role_balance': float(role_engine.compute_team_role_balance([agent_lookup.get(p, 'Unknown') for p in resolved_players])),
        'format_modifier': 1.0 if format == 'Bo1' else 1.15 if format == 'Bo3' else 1.25,
        'stage_modifier': 1.2 if stage == 'Playoffs' else 1.0
    }
    
    if team_features['mech_mean'] > 31:
        team_features['chemistry'], team_features['role_balance'] = 0.95, 100.0
    elif team_features['mech_mean'] > 28:
        team_features['chemistry'] = round(0.5 * team_features['chemistry'] + 0.45, 4)
        team_features['role_balance'] = round(0.5 * team_features['role_balance'] + 45.0, 2)

    return pred_engine.apply_modifiers(team_features, fmt=format, stage=stage)


def player_matchup_analysis(team_a: list, team_b: list, map_name: str) -> dict:
    canonical_map = normalize_map_name(map_name)
    analysis = []
    
    for pa_name, pb_name in zip(team_a, team_b):
        stats_a, stats_b = _get_player_stats(pa_name), _get_player_stats(pb_name)
        base_a, base_b = stats_a['mech_mean'], stats_b['mech_mean']
        
        map_adj_a = map_engine.get_player_map_score(pa_name, canonical_map)['map_score'] / 100.0
        map_adj_b = map_engine.get_player_map_score(pb_name, canonical_map)['map_score'] / 100.0
        
        final_a, final_b = round(base_a + (map_adj_a * 10), 2), round(base_b + (map_adj_b * 10), 2)
        advantage = 'E' if final_a == final_b else 'A' if final_a > final_b else 'B'
        
        analysis.append({
            'player_a': pa_name, 'player_b': pb_name,
            'map_score_a': final_a, 'map_score_b': final_b, 'advantage': advantage
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

    # --- FINAL SUPERTEAM BLEND (Soft Override) ---
    # We no longer 'floor' the win prob. Instead, we use a weighted blend.
    # This ensures map-to-map variance stays visible.
    skill_diff = fa['mech_mean'] - fb['mech_mean']
    
    # Correction target based on skill gap
    skill_bias = 0.0
    if skill_diff > 1.0:
        skill_bias = min(0.45, 0.15 + (skill_diff * 0.05))
    elif skill_diff < -1.0:
        skill_bias = max(-0.45, -0.15 + (skill_diff * 0.05))

    for res in map_results:
        # Blend: 70% Map-Specific Logic + 30% Skill-Correction Bias
        orig_a = float(res['team_a_win_prob']) / 100.0
        blended_a = (0.7 * orig_a) + (0.3 * (0.5 + skill_bias))
        
        res['team_a_win_prob'] = round(blended_a * 100, 3)
        res['team_b_win_prob'] = round(100.0 - res['team_a_win_prob'], 3)
        res['predicted_winner'] = 'Team A' if blended_a >= 0.5 else 'Team B'

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
            m: round(map_engine.get_player_map_score(resolved, m)['map_score'] * max(1.0, float(fix['mech_mean']) / 22.0), 3)
            for m in (SUPPORTED_MAPS if SUPPORTED_MAPS else ['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze'])
        }
    }
    return res


# Module 1: current roster forecasting endpoint (tier1_rosters.csv source + fallback hardcoded)


class AIStrategicAnalyst:
    """
    Agentic Layer: Provides 'Human-Like' strategic reasoning and brand 
    reputation analysis to balance the raw ML numbers.
    """
    
    @staticmethod
    def calculate_tier(avg_mech: float, chemistry: float, players: List[str]) -> str:
        """
        Dynamic Tiering: No more hardcoded 'Kings'.
        Tiers are EARNED by the roster potential in 2026.
        """
        # S-Tier: Elite Core (Mechanical giants with championship chemistry)
        if avg_mech >= 41.5 and chemistry >= 0.85:
            return 'S'
            
        # A-Tier: Top Tier Firepower (High mechanical skill but perhaps less history)
        if avg_mech >= 38.0:
            return 'A'
            
        # B-Tier: Mid-Tier/Unstable (Good pros but missing elite firepower/chemistry)
        if avg_mech >= 34.0:
            return 'B'
            
        # C-Tier: Neutral/Rising (Standard pro baseline or new rosters)
        return 'C'

    @staticmethod
    def get_expert_verdict(team_a: str, team_b: str, fa: Dict, fb: Dict) -> Dict:
        # Calculate Tiers dynamically based on the stats in the request
        tier_a = AIStrategicAnalyst.calculate_tier(fa['mech_mean'], fa['chemistry'], [])
        tier_b = AIStrategicAnalyst.calculate_tier(fb['mech_mean'], fb['chemistry'], [])
        
        # 1. TIER BIAS (The Reputation Factor)
        # S=1.05, A=1.025, B=1.0, C=0.98 (Dampened for 2026 Neutrality)
        multiplier = {'S': 1.05, 'A': 1.025, 'B': 1.0, 'C': 0.98}
        res_a = multiplier[tier_a]
        res_b = multiplier[tier_b]
        
        # 2. CHEMISTRY REASONING (Logic over Stats)
        chem_a = fa.get('chemistry', 0.6)
        chem_b = fb.get('chemistry', 0.6)
        
        # 3. VERDICT GENERATION
        expert_win_prob_a = 0.5 + (res_a - res_b) + (chem_a - chem_b) * 0.15
        expert_win_prob_a = max(min(expert_win_prob_a, 0.70), 0.30)
        
        # 4. ANALYST NOTE (The Conclusion)
        note = ""
        if tier_a == 'S' and tier_b != 'S':
            verdict = f"Analyst Verdict: {team_a} has successfully built an S-Tier 'Superteam' for the 2026 season. Their synergy and raw firepower outclass the {team_b} roster."
        elif chem_a > chem_b + 0.15:
            verdict = f"Neutrality Factor: The power balance has shifted. {team_a} holds a massive strategic advantage due to their superior chemistry in this 'Neutral' 2026 era."
        elif abs(chem_a - chem_b) < 0.05:
            verdict = f"Final Verdict: Perfectly balanced 2026 matchup. Neither team holds a 'Legacy King' advantage; the series will likely be decided by individual map heroics."
        else:
            verdict = f"A realistic 2026 estimation: The AI Analyst identifies {team_a} as the marginal favorite based on their slightly more disciplined 5-man core."

        # Real-World Consistency Check: 
        # Even with a stats landslide, no pro squad has < 15% chance in Bo3.
        # 15% indicates 'Absolute Domination' by the opponent.
        expert_a = max(0.15, min(0.85, expert_win_prob_a))
        return {
            'expert_win_prob_a': round(expert_a * 100, 2),
            'expert_win_prob_b': round((1.0 - expert_a) * 100, 2),
            'verdict': verdict,
            'team_a_tier': tier_a,
            'team_b_tier': tier_b
        }


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
    
    team_a_roster = normalized[team_a_key][:5]
    team_b_roster = normalized[team_b_key][:5]

    req = MatchPredictRequest(
        team_a=team_a_roster,
        team_b=team_b_roster,
        map_pool=map_pool,
        format=format,
        stage=stage
    )
    
    # 1. RAW NUMBERS LAYER (The Pretty Stats)
    prediction = predict_match(req)
    
    # 2. AGENTIC AI LAYER (The Real World Conclusion)
    # Re-calculate features briefly for the analyst
    fa_mock = build_team_features(team_a_roster, map_pool[0], stage, format)
    fb_mock = build_team_features(team_b_roster, map_pool[0], stage, format)
    expert = AIStrategicAnalyst.get_expert_verdict(team_a, team_b, fa_mock, fb_mock)
    
    # 3. HYBRID BLEND (50/50 Consensus)
    ml_a = prediction['team_a_average_win_prob'] / 100.0
    agent_a = float(expert['expert_win_prob_a']) / 100.0
    
    # Dynamic Confidence: If ML is extreme, trust Agentic more to avoid 2% errors.
    weight_ml = 0.5
    if ml_a > 0.9 or ml_a < 0.1:
        weight_ml = 0.3 # Dampen the ML radicalism
        
    hybrid_a = (ml_a * weight_ml) + (agent_a * (1.0 - weight_ml))
    hybrid_b = 1.0 - hybrid_a

    return {
        **prediction,
        'team_a': team_a,
        'team_b': team_b,
        'team_a_average_win_prob': round(hybrid_a * 100, 2),
        'team_b_average_win_prob': round(hybrid_b * 100, 2),
        'ai_analyst_verdict': expert['verdict'],
        'team_a_tier': expert['team_a_tier'],
        'team_b_tier': expert['team_b_tier'],
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
