import os
import sys
import pickle
import pandas as pd
import numpy as np

# Add project root to sys.path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer
from src.scoring.map_score import MapScoreEngine
from src.scoring.chemistry import ChemistryEngine

def export_production_data():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    OUTPUT_PATH = os.path.join(MODELS_DIR, 'vct_production.all.pkl')

    print("--- VALORANT AI: Production Data Exporter ---")
    print(f"Ingesting raw data from: {DATA_DIR}")

    loader = VCTDataLoader(data_dir=DATA_DIR)
    ov = loader.load_overviews()
    kills = loader.load_kills_stats()
    maps_scores = loader.load_maps_scores()

    if ov.empty or kills.empty or maps_scores.empty:
        print("[ERROR] Could not load raw data. verify data/raw exists.")
        return

    cleaner = VCTCleaner()
    # Logic matching backend/api.py ingestion
    base_join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Year']
    join_cols = [c for c in base_join_cols if c in ov.columns and c in kills.columns]
    
    print(f"Merging {len(ov)} overview records with kills stats...")
    merged = pd.merge(ov, kills, on=join_cols, how='left', suffixes=('', '_kills'))
    if 'Agents' not in merged.columns and 'Agents_x' in merged.columns:
        merged['Agents'] = merged['Agents_x']
    
    clean_df = cleaner.clean_overviews(merged)
    
    print("Computing Player Scores (Mechanical, Clutch, Efficiency)...")
    scorer = PlayerScorer()
    scored_df = scorer.compute_overall_score(clean_df)
    
    # 1. Player Summary
    print("Generating Player Summaries...")
    player_summary = scored_df.groupby('Player').agg(
        mech_mean=('Mechanical_Score', 'mean'),
        clutch_mean=('Clutch_Score', 'mean'),
        entry_mean=('Entry_Success', 'mean'),
        util_mean=('Utility_Score', 'mean'),
        eco_mean=('Economic_Score', 'mean'),
        consistency_mean=('Consistency_Score', 'mean')
    ).to_dict('index')

    # 2. Map Lookups
    print("Building Map Performance Engine...")
    map_engine = MapScoreEngine()
    map_engine.build(clean_df, maps_scores)
    map_lookup = map_engine._player_map_table.to_dict('index')

    # 3. Chemistry History
    print("Building Pairwise Chemistry History...")
    chem_engine = ChemistryEngine(clean_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
    chem_history = chem_engine._pair_history

    # 4. Agent Lookup (Preferred Agent)
    print("Determining Comfort Agents...")
    agent_lookup = (
        clean_df.groupby('Player')['Agents']
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
        .to_dict()
    )

    # 5. Metadata
    elite_thresh = float(scored_df.groupby('Player')['Mechanical_Score'].mean().quantile(0.95))

    package = {
        'player_summary': player_summary,
        'map_lookup': map_lookup,
        'chemistry_history': chem_history,
        'agent_lookup': agent_lookup,
        'metadata': {
            'elite_quantile_95': elite_thresh,
            'exported_at': pd.Timestamp.now().isoformat()
        }
    }

    print(f"Exporting to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(package, f)

    print(f"Successfully exported production package with {len(player_summary)} players.")

if __name__ == "__main__":
    export_production_data()
