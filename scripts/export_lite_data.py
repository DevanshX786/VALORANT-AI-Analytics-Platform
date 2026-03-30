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

def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"--- Production-Lite Sync Engine ---")
    print(f"Loading 1.3GB raw VCT dataset from: {DATA_DIR}")
    
    loader = VCTDataLoader(data_dir=DATA_DIR)
    ov = loader.load_overviews()
    kills = loader.load_kills_stats()
    maps_scores = loader.load_maps_scores()
    
    print("Cleaning and Scoring (This may take a few seconds)...")
    cleaner = VCTCleaner()
    base_join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Year']
    join_cols = [c for c in base_join_cols if c in ov.columns and c in kills.columns]
    merged = pd.merge(ov, kills, on=join_cols, how='left', suffixes=('', '_kills'))
    
    if 'Agents' not in merged.columns:
        if 'Agents_x' in merged.columns: merged['Agents'] = merged['Agents_x']
    
    clean_df = cleaner.clean_overviews(merged)
    scorer = PlayerScorer()
    scored_df = scorer.compute_overall_score(clean_df)
    
    # 1. Export Player Summary
    print("Exporting player_summary.pkl...")
    player_summary = scored_df.groupby('Player').agg(
        mech_mean=('Mechanical_Score', 'mean'),
        clutch_mean=('Clutch_Score', 'mean'),
        entry_mean=('Entry_Success', 'mean'),
        util_mean=('Utility_Score', 'mean'),
        eco_mean=('Economic_Score', 'mean'),
        consistency_mean=('Consistency_Score', 'mean')
    ).reset_index().set_index('Player')
    
    with open(os.path.join(MODELS_DIR, 'player_summary.pkl'), 'wb') as f:
        pickle.dump(player_summary, f)

    # 2. Export Map Engine State
    print("Exporting map_engine.pkl...")
    map_engine = MapScoreEngine()
    map_engine.build(clean_df, maps_scores)
    
    with open(os.path.join(MODELS_DIR, 'map_engine.pkl'), 'wb') as f:
        pickle.dump(map_engine, f)

    # 3. Export Chemistry Engine State
    print("Exporting chemistry_engine.pkl...")
    from src.scoring.chemistry import ChemistryEngine
    chem_engine = ChemistryEngine(clean_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
    
    with open(os.path.join(MODELS_DIR, 'chemistry_engine.pkl'), 'wb') as f:
        pickle.dump(chem_engine, f)

    print(f"--- SUCCESS ---")
    print(f"Files created in {MODELS_DIR}. Please push these to GitHub for Render deployment.")

if __name__ == "__main__":
    main()
