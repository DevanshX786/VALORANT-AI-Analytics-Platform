import os
import sys
import pandas as pd
import pickle

# Add project root to sys.path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer
from src.scoring.map_score import MapScoreEngine
from src.scoring.chemistry import ChemistryEngine

def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"--- Ultra-Lite Production Consolidation ---")
    print(f"Ingesting 1.3GB dataset for final optimization...")
    
    loader = VCTDataLoader(data_dir=DATA_DIR)
    ov = loader.load_overviews()
    kills = loader.load_kills_stats()
    maps_scores = loader.load_maps_scores()
    
    cleaner = VCTCleaner()
    base_join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Year']
    join_cols = [c for c in base_join_cols if c in ov.columns and c in kills.columns]
    merged = pd.merge(ov, kills, on=join_cols, how='left', suffixes=('', '_kills'))
    
    if 'Agents' not in merged.columns:
        if 'Agents_x' in merged.columns: merged['Agents'] = merged['Agents_x']
    
    clean_df = cleaner.clean_overviews(merged)
    scorer = PlayerScorer()
    scored_df = scorer.compute_overall_score(clean_df)
    
    # 1. Player Summary Dictionary
    print("Formatting player_summary...")
    player_summary_df = scored_df.groupby('Player').agg(
        mech_mean=('Mechanical_Score', 'mean'),
        clutch_mean=('Clutch_Score', 'mean'),
        entry_mean=('Entry_Success', 'mean'),
        util_mean=('Utility_Score', 'mean'),
        eco_mean=('Economic_Score', 'mean'),
        consistency_mean=('Consistency_Score', 'mean')
    ).reset_index()
    player_summary_dict = player_summary_df.set_index('Player').to_dict('index')

    # 2. Map Engine Dictionary (FLATTEN TO DICT)
    print("Formatting map_engine (flattening to lookup dict)...")
    map_engine = MapScoreEngine()
    map_engine.build(clean_df, maps_scores)
    # Convert multi-index DataFrame to dict: {(player, map): {metrics}}
    map_lookup = map_engine._player_map_table.to_dict('index')

    # 3. Chemistry Engine Pair History
    print("Formatting chemistry_engine...")
    chem_engine = ChemistryEngine(clean_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
    chem_history = chem_engine._pair_history

    # 4. Agent Lookup
    print("Formatting agent_lookup...")
    agent_lookup = (
        clean_df.groupby('Player')['Agents']
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
        .to_dict()
    )

    # 5. CONSOLIDATE EVERYTHING
    print("Consolidating to vct_production_data.all.pkl...")
    production_package = {
        'player_summary': player_summary_dict,
        'map_lookup': map_lookup,
        'chemistry_history': chem_history,
        'agent_lookup': agent_lookup,
        'metadata': {
            'player_count': len(player_summary_dict),
            'map_pair_count': len(map_lookup),
            'chem_pair_count': len(chem_history),
            'elite_quantile_95': float(player_summary_df['mech_mean'].quantile(0.95))
        }
    }

    with open(os.path.join(MODELS_DIR, 'vct_production.all.pkl'), 'wb') as f:
        pickle.dump(production_package, f)

    print(f"--- SUCCESS ---")
    print(f"FINAL CONSOLIDATED PACKAGE CREATED: models/vct_production.all.pkl")
    print(f"Replace your Render push with this single file.")

if __name__ == "__main__":
    main()
