# %% [markdown]
# # Step 4 Prototyping: XGBoost Baseline Match Prediction Model
# We aggregate individual player scores into team-level features,
# then build a head-to-head diff row per match to train a Win/Loss classifier.

# %%
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner
from src.scoring.individual_score import PlayerScorer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import xgboost as xgb

# %%
# -----------------------------------------------------------------------
# STEP A: Load, Clean, Score
# -----------------------------------------------------------------------
print("Loading data (all years)...")
loader = VCTDataLoader(data_dir=os.path.join(os.path.dirname(__file__), '../data/raw'))
overview_df  = loader.load_overviews()
kills_df     = loader.load_kills_stats()
scores_df    = loader.load_scores()

# Merge kills stats into overview for multi-kill/clutch columns
join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Agents', 'Year']
merged_df = pd.merge(overview_df, kills_df, on=join_cols, how='inner')

cleaner   = VCTCleaner()
clean_df  = cleaner.clean_overviews(merged_df)

print(f"Clean records: {len(clean_df)}")

# %%
# Compute individual scores per player per map
scorer    = PlayerScorer()
scored_df = scorer.compute_overall_score(clean_df)
print("Player scores computed.")

# %%
# -----------------------------------------------------------------------
# STEP B: Aggregate to Team Level per Match
# -----------------------------------------------------------------------
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
    )
    .reset_index()
)
team_agg['mech_std'] = team_agg['mech_std'].fillna(0)
print(f"Team-level aggregation: {len(team_agg)} team-match rows")

# %%
# -----------------------------------------------------------------------
# STEP C: Build Head-to-Head Match Feature Rows
# -----------------------------------------------------------------------
# We need scores.csv columns: Match Name, Team A, Team B, Team A Score, Team B Score
# Clean scores_df
scores_df = scores_df[scores_df['Team A'] != 'TBD']
scores_df = scores_df[scores_df['Team B'] != 'TBD']
scores_df = scores_df.dropna(subset=['Team A Score', 'Team B Score'])
scores_df['Team A Score'] = pd.to_numeric(scores_df['Team A Score'], errors='coerce')
scores_df['Team B Score'] = pd.to_numeric(scores_df['Team B Score'], errors='coerce')
scores_df = scores_df.dropna(subset=['Team A Score', 'Team B Score'])
scores_df['label'] = (scores_df['Team A Score'] > scores_df['Team B Score']).astype(int)

print(f"Match labels: {len(scores_df)} total | {scores_df['label'].mean()*100:.1f}% Team A wins")

# Helper: get team feature row from aggregated table
feature_cols = ['mech_mean', 'mech_max', 'mech_std', 'clutch_sum', 'entry_mean', 'util_mean', 'eco_mean']

def get_team_features(match_name, team_name):
    row = team_agg[(team_agg['Match Name'] == match_name) & (team_agg['Team'] == team_name)]
    if row.empty:
        return None
    return row[feature_cols].iloc[0].values

# Build diffs
rows = []
for _, match in scores_df.iterrows():
    fa = get_team_features(match['Match Name'], match['Team A'])
    fb = get_team_features(match['Match Name'], match['Team B'])
    if fa is None or fb is None:
        continue
    diff = fa - fb
    rows.append(list(diff) + [match['label']])

diff_cols = [f'{c}_diff' for c in feature_cols] + ['label']
model_df  = pd.DataFrame(rows, columns=diff_cols)
print(f"Built {len(model_df)} head-to-head match rows for training.")

# %%
# -----------------------------------------------------------------------
# STEP D: Train XGBoost Baseline
# -----------------------------------------------------------------------
X = model_df.drop(columns=['label'])
y = model_df['label']

# Chronological split: first 80% of rows = train, last 20% = test (no data leakage)
split_idx = int(len(model_df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# %%
# -----------------------------------------------------------------------
# STEP E: Evaluate
# -----------------------------------------------------------------------
y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
ll   = log_loss(y_test, y_prob)

print("\n========== BASELINE MODEL RESULTS ==========")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Log Loss : {ll:.4f}")
print("=============================================\n")

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:")
for feat, score in importance.items():
    print(f"  {feat:<30} {score:.4f}")

# Save model
save_path = os.path.join(os.path.dirname(__file__), '../models/baseline_xgb.json')
model.save_model(save_path)
print(f"\nModel saved to {save_path}")
