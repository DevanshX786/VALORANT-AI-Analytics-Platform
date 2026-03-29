import os
import pandas as pd
import numpy as np
from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer
from src.scoring.map_score import MapScoreEngine
from src.models.baseline_model import BaselineModel

print('Loading data...')
project_dir = os.path.abspath(os.path.dirname(__file__))
raw_dir = os.path.join(project_dir, 'data', 'raw')
loader = VCTDataLoader(data_dir=raw_dir)

ov = loader.load_overviews()
kills = loader.load_kills_stats()
map_scores = loader.load_maps_scores()
scores = loader.load_scores()

if 'Year' not in ov.columns:
    ov['Year'] = 'unknown'

cleaner = VCTCleaner()
merged = pd.merge(ov, kills, on=['Match Name', 'Map', 'Player', 'Team', 'Agents', 'Year'], how='inner')
clean_df = cleaner.clean_overviews(merged)

scorer = PlayerScorer()
scored_df = scorer.compute_overall_score(clean_df)

map_engine = MapScoreEngine()
map_engine.build(clean_df, map_scores)

print('Preparing model and team features...')
model = BaselineModel()
model_path = os.path.join(project_dir, 'models', 'baseline_xgb.json')
if os.path.exists(model_path):
    model.load(model_path)
else:
    print('Model file not found, training new model...')

team_agg = model.build_team_features(scored_df, map_score_engine=map_engine)
model_df = model.build_match_dataset(team_agg, scores)

print(f'Dataset rows for training/eval: {len(model_df)}')

X_test, y_test = model.train(model_df)
metrics = model.evaluate(X_test, y_test)
print('Evaluation metrics:', metrics)
