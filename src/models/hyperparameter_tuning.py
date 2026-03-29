import os
import sys
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer
from src.scoring.map_score import MapScoreEngine
from src.models.baseline_model import BaselineModel


def build_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    loader = VCTDataLoader(data_dir=data_dir)
    overview_df = loader.load_overviews()
    kills_df    = loader.load_kills_stats()
    scores_df   = loader.load_scores()

    join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Agents', 'Year']
    merged_df = pd.merge(overview_df, kills_df, on=join_cols, how='inner')

    cleaner = VCTCleaner()
    clean_df = cleaner.clean_overviews(merged_df)

    scorer = PlayerScorer()
    scored_df = scorer.compute_overall_score(clean_df)

    # Build map score from Step 4 and include in team-level features
    map_engine = MapScoreEngine()
    map_engine.build(clean_df, loader.load_maps_scores())

    pipeline = BaselineModel()
    team_agg = pipeline.build_team_features(scored_df, map_score_engine=map_engine)
    model_df = pipeline.build_match_dataset(team_agg, scores_df)
    return model_df


def tune_model(model_df):
    X = model_df.drop(columns=['label'])
    y = model_df['label']

    split_idx = int(len(model_df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'n_estimators': [150, 200, 250, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
    }

    best = {'score': 0.0, 'params': None, 'metrics': None}

    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            **params
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        ll = log_loss(y_test, y_prob)

        if auc > best['score']:
            best = {
                'score': auc,
                'params': params,
                'metrics': {
                    'accuracy': acc,
                    'auc_roc': auc,
                    'log_loss': ll,
                },
            }

        print(f"params={params} acc={acc:.4f} auc={auc:.4f} logloss={ll:.4f}")

    print("\nBEST SETTINGS")
    print(best)
    return best


if __name__ == '__main__':
    print("Building dataset...")
    model_df = build_dataset()
    print(f"Dataset rows: {len(model_df)}")
    print("Running hyperparameter grid search...")
    best_config = tune_model(model_df)
    print("Done.")
    print(best_config)
