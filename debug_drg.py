import pandas as pd
from backend.api import build_team_features, map_engine, chem_engine, role_engine
import json

rosters = {"Dragon Ranger Gaming": ["vo0kashu", "flex1n", "nicc", "spiritZ1", "akeman"], "Envy": ["rossy", "keznit", "p0ppin", "demon1", "eggsterr"]}

fA = build_team_features(rosters["Dragon Ranger Gaming"], "Bind", "Group", "Bo3")
fB = build_team_features(rosters["Envy"], "Bind", "Group", "Bo3")

import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model("models/baseline_xgb.json")

FEATURE_COLS = [
    'mech_mean', 'mech_max', 'mech_std',
    'clutch_sum', 'entry_mean', 'util_mean', 'eco_mean',
    'consistency', 'chemistry', 'map_score', 'role_balance'
]

diff = {f'{k}_diff': fA[k] - fB[k] for k in FEATURE_COLS}
row = pd.DataFrame([diff])
prob_a = model.predict_proba(row)[0, 1]

print("\n--- Features ---")
for k in FEATURE_COLS:
    print(f"{k}: DRG={fA[k]:.2f} Envy={fB[k]:.2f}  Diff={diff[f'{k}_diff']:.2f}")

print("\nWin Prob DRG:", prob_a)

importance = pd.Series(model.feature_importances_, index=[f'{k}_diff' for k in FEATURE_COLS]).sort_values(ascending=False)
print("\n--- Feature Importance ---")
print(importance)
