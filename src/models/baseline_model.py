import os
import sys
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings('ignore')

# Ensure src/ is importable when run directly
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer
from src.scoring.role_balance import RoleBalanceEngine
from src.scoring.chemistry import ChemistryEngine
from src.scoring.map_score import MapScoreEngine
from src.models.prediction_engine import PredictionEngine


FEATURE_COLS = [
    'mech_mean', 'mech_max', 'mech_std',
    'clutch_sum', 'entry_mean', 'util_mean', 'eco_mean',
    'consistency', 'chemistry', 'map_score', 'role_balance'
]


class BaselineModel:
    """
    End-to-end pipeline that:
      1. Aggregates per-player scores into team-level features.
      2. Builds head-to-head match diff rows with Win/Loss labels.
      3. Trains an XGBoost classifier using a chronological 80/20 split.
      4. Evaluates and saves the trained model.
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        self.feature_cols = [f'{c}_diff' for c in FEATURE_COLS]

    # ------------------------------------------------------------------
    # Step A: Aggregate 5 players → one team row per match
    # ------------------------------------------------------------------
    def build_team_features(self, scored_df: pd.DataFrame, map_score_engine: MapScoreEngine = None) -> pd.DataFrame:
        """
        Groups scored player rows by (Match Name, Team, Year) and
        produces aggregated team-level statistics.
        """
        # optionally compute per-player map score with map engine
        if map_score_engine is not None and 'Player' in scored_df.columns and 'Map' in scored_df.columns:
            scored_df = scored_df.copy()
            scored_df['player_map_score'] = scored_df.apply(
                lambda row: map_score_engine.get_player_map_score(row['Player'], row['Map'])['map_score'],
                axis=1
            )
        else:
            scored_df['player_map_score'] = 50.0

        team_agg = (
            scored_df
            .groupby(['Match Name', 'Team', 'Year'])
            .agg(
                mech_mean  = ('Mechanical_Score', 'mean'),
                mech_max   = ('Mechanical_Score', 'max'),
                mech_std   = ('Mechanical_Score', 'std'),
                clutch_sum = ('Clutch_Score',     'sum'),
                entry_mean = ('Entry_Success',    'mean'),
                util_mean  = ('Utility_Score',    'mean'),
                eco_mean   = ('Economic_Score',   'mean'),
                map_score  = ('player_map_score', 'mean'),
            )
            .reset_index()
        )
        team_agg['mech_std'] = team_agg['mech_std'].fillna(0)

        # Step 6: Add consistency (team-level mean of player consistency scores)
        consistency_df = (
            scored_df
            .groupby(['Match Name', 'Team'])['Consistency_Score']
            .mean()
            .reset_index()
            .rename(columns={'Consistency_Score':'consistency'})
        )
        team_agg = pd.merge(team_agg, consistency_df, on=['Match Name', 'Team'], how='left')
        team_agg['consistency'] = team_agg['consistency'].fillna(5.0)

        # Step 3: Team chemistry score
        chem_engine = ChemistryEngine(scored_df[['Match Name', 'Team', 'Player', 'Year']].drop_duplicates())
        roster_df = (
            scored_df[['Match Name', 'Team', 'Player']]
            .drop_duplicates()
            .groupby(['Match Name', 'Team'])['Player']
            .apply(list)
            .reset_index()
        )
        roster_df['chemistry'] = roster_df['Player'].apply(lambda players: chem_engine.get_team_chemistry_score(players))

        team_agg = pd.merge(team_agg, roster_df[['Match Name', 'Team', 'chemistry']], on=['Match Name', 'Team'], how='left')
        team_agg['chemistry'] = team_agg['chemistry'].fillna(0.5)

        # Step 4: Map Score calculation already natively merged during baseline aggregation on line 78.
        # Step 5: Role balance score based on agents in the roster
        role_engine = RoleBalanceEngine()
        role_balance_df = role_engine.team_role_balance_from_df(scored_df)

        team_agg = pd.merge(
            team_agg,
            role_balance_df,
            on=['Match Name', 'Team'],
            how='left'
        )

        team_agg['Role_Balance_Score'] = team_agg['Role_Balance_Score'].fillna(50.0)
        team_agg.rename(columns={'Role_Balance_Score': 'role_balance'}, inplace=True)

        return team_agg

    # ------------------------------------------------------------------
    # Step B: Build head-to-head diff rows with labels
    # ------------------------------------------------------------------
    def build_match_dataset(
        self,
        team_agg: pd.DataFrame,
        scores_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each match in scores_df, subtracts Team B features from Team A features.
        Returns a DataFrame with one row per match and a binary label (1 = Team A won).
        """
        # Clean scores
        scores = scores_df[
            (scores_df['Team A'] != 'TBD') & (scores_df['Team B'] != 'TBD')
        ].copy()
        scores['Team A Score'] = pd.to_numeric(scores['Team A Score'], errors='coerce')
        scores['Team B Score'] = pd.to_numeric(scores['Team B Score'], errors='coerce')
        scores = scores.dropna(subset=['Team A Score', 'Team B Score'])
        scores['label'] = (scores['Team A Score'] > scores['Team B Score']).astype(int)

        # Index team_agg for fast lookup
        team_index = team_agg.set_index(['Match Name', 'Team'])

        rows = []
        for _, match in scores.iterrows():
            mn = match['Match Name']
            fmt = match.get('Format', 'Bo3') if 'Format' in match else 'Bo3'
            stage = match.get('Stage', 'Group') if 'Stage' in match else 'Group'
            try:
                row_a = team_index.loc[(mn, match['Team A'])]
                row_b = team_index.loc[(mn, match['Team B'])]

                if isinstance(row_a, pd.DataFrame):
                    row_a = row_a.iloc[0]
                if isinstance(row_b, pd.DataFrame):
                    row_b = row_b.iloc[0]

                pred_engine = PredictionEngine(match_format=fmt, stage=stage)
                team_a_features = pred_engine.apply_modifiers(row_a[FEATURE_COLS].to_dict(), fmt, stage)
                team_b_features = pred_engine.apply_modifiers(row_b[FEATURE_COLS].to_dict(), fmt, stage)

                fa = np.array([team_a_features[c] for c in FEATURE_COLS], dtype=float)
                fb = np.array([team_b_features[c] for c in FEATURE_COLS], dtype=float)
            except KeyError:
                continue
            rows.append(list(fa - fb) + [match['label']])


        diff_cols = self.feature_cols + ['label']
        return pd.DataFrame(rows, columns=diff_cols)

    # ------------------------------------------------------------------
    # Step C: Train (chronological split — no data leakage)
    # ------------------------------------------------------------------
    def train(self, model_df: pd.DataFrame) -> tuple:
        """
        Trains the XGBoost classifier on the provided match dataset.
        Uses first 80% of rows as train, last 20% as test (chronological order assumed).
        Returns (X_test, y_test) for evaluation.
        """
        X = model_df.drop(columns=['label'])
        y = model_df['label']

        split_idx = int(len(model_df) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Training on {len(X_train)} matches | Testing on {len(X_test)} matches...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        return X_test, y_test

    # ------------------------------------------------------------------
    # Step D: Evaluate
    # ------------------------------------------------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Prints and returns accuracy, AUC-ROC, and log-loss metrics.
        """
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc':  roc_auc_score(y_test, y_prob),
            'log_loss': log_loss(y_test, y_prob),
        }

        print("\n========== BASELINE MODEL RESULTS ==========")
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
        print(f"  Log Loss : {metrics['log_loss']:.4f}")
        print("=============================================\n")

        importance = pd.Series(
            self.model.feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=False)
        print("Feature Importances:")
        for feat, score in importance.items():
            print(f"  {feat:<35} {score:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Step E: Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str = "models/baseline_xgb.json"):
        """Saves the trained XGBoost model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save_model(path)
        print(f"\nModel saved -> {path}")

    def load(self, path: str = "models/baseline_xgb.json"):
        """Loads a previously saved XGBoost model from disk."""
        self.model.load_model(path)
        print(f"Model loaded <- {path}")

    # ------------------------------------------------------------------
    # Convenience: predict a single match head-to-head
    # ------------------------------------------------------------------
    def predict_match(self, team_a_features: dict, team_b_features: dict) -> dict:
        """
        Given two dicts of team feature values (mech_mean, mech_max, etc.),
        returns the predicted winner and win probability for Team A.
        """
        diff = {f'{k}_diff': team_a_features[k] - team_b_features[k] for k in FEATURE_COLS}
        row = pd.DataFrame([diff])
        prob_a = float(self.model.predict_proba(row)[0, 1])
        return {
            'team_a_win_prob': round(prob_a, 4),
            'team_b_win_prob': round(1 - prob_a, 4),
            'predicted_winner': 'Team A' if prob_a >= 0.5 else 'Team B'
        }


# -----------------------------------------------------------------------
# Run as standalone script to train + save
# -----------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')

    print("Loading data...")
    loader = VCTDataLoader(data_dir=data_dir)
    overview_df = loader.load_overviews()
    kills_df    = loader.load_kills_stats()
    scores_df   = loader.load_scores()
    maps_scores = loader.load_maps_scores()

    join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Agents', 'Year']
    merged_df = pd.merge(overview_df, kills_df, on=join_cols, how='inner')

    cleaner   = VCTCleaner()
    clean_df  = cleaner.clean_overviews(merged_df)

    scorer    = PlayerScorer()
    scored_df = scorer.compute_overall_score(clean_df)

    # Map score engine from Step 4 – used to generate map_score feature per player-team
    map_engine = MapScoreEngine()
    map_engine.build(clean_df, maps_scores)

    pipeline  = BaselineModel()
    team_agg  = pipeline.build_team_features(scored_df, map_score_engine=map_engine)
    model_df  = pipeline.build_match_dataset(team_agg, scores_df)
    print(f"Dataset: {len(model_df)} match rows")

    X_test, y_test = pipeline.train(model_df)
    pipeline.evaluate(X_test, y_test)
    pipeline.save("models/baseline_xgb.json")
