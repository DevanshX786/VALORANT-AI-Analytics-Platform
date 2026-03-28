# %% [markdown]
# # Step 3 Prototyping: Individual Player Scoring
# This script dynamically tests the 7-pillar model on 2024 VCT matches using our data pipeline.

# %%
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # ignore DtypeWarning for low_memory

# ensure src is in path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.loader import VCTDataLoader
from src.data.cleaner import VCTCleaner

# %%
print("Loading core statistics...")
loader = VCTDataLoader(data_dir=os.path.join(os.path.dirname(__file__), '../data/raw'))
overview_df = loader.load_overviews()
kills_stats_df = loader.load_kills_stats()

print(f"Loaded {len(overview_df)} match overview rows and {len(kills_stats_df)} kill stats rows.")

# Merge on composite keys to bring '2k', '3k', 'Spike Plants' into the overview
join_cols = ['Match Name', 'Map', 'Player', 'Team', 'Agents', 'Year']
merged_df = pd.merge(overview_df, kills_stats_df, on=join_cols, how='inner')
print(f"Joined table contains {len(merged_df)} complete player-map records.")

cleaner = VCTCleaner()
clean_df = cleaner.clean_overviews(merged_df)

# Prototyping strictly on the latest 2024 season
df_2024 = clean_df[clean_df['Year'] == '2024'].copy()
print(f"Prototyping Scoring Engine on {len(df_2024)} rows for VCT 2024 data.\n")

# %% [markdown]
# ## 1. Mechanical Score
# logic: (ACS * 0.45) + ((Kills / max(Deaths, 1)) * 30) + ((2k*2 + 3k*3 + 4k*4 + 5k*5) * 2)

# %%
df_2024['Mechanical_Score'] = (
    pd.to_numeric(df_2024['Average Combat Score'], errors='coerce').fillna(0) * 0.45 +
    (pd.to_numeric(df_2024['Kills'], errors='coerce').fillna(0) / 
     pd.to_numeric(df_2024['Deaths'], errors='coerce').replace(0, 1).fillna(1)) * 30 +
    ((pd.to_numeric(df_2024['2k'], errors='coerce').fillna(0) * 2) + 
     (pd.to_numeric(df_2024['3k'], errors='coerce').fillna(0) * 3) + 
     (pd.to_numeric(df_2024['4k'], errors='coerce').fillna(0) * 4) + 
     (pd.to_numeric(df_2024['5k'], errors='coerce').fillna(0) * 5)) * 2
)

# %% [markdown]
# ## 2. Clutch Score
# logic: (1v1 * 1) + (1v2 * 2) + etc...

# %%
df_2024['Clutch_Score'] = (
    (pd.to_numeric(df_2024['1v1'], errors='coerce').fillna(0) * 1) +
    (pd.to_numeric(df_2024['1v2'], errors='coerce').fillna(0) * 2) +
    (pd.to_numeric(df_2024['1v3'], errors='coerce').fillna(0) * 4) +
    (pd.to_numeric(df_2024['1v4'], errors='coerce').fillna(0) * 8) +
    (pd.to_numeric(df_2024['1v5'], errors='coerce').fillna(0) * 15)
)

# %% [markdown]
# ## 3. Entry Success
# logic: First Kills / max((First Kills + First Deaths), 1) * 100 + (First Kills - Deaths)

# %%
fk = pd.to_numeric(df_2024['First Kills'], errors='coerce').fillna(0)
fd = pd.to_numeric(df_2024['First Deaths'], errors='coerce').fillna(0)
fkd = pd.to_numeric(df_2024['Kills - Deaths (FKD)'], errors='coerce').fillna(0)

# FK / Total First Duels
fk_ratio = fk / (fk + fd).replace(0, 1)

df_2024['Entry_Success'] = (fk_ratio * 100) + fkd

# %% [markdown]
# ## Output Validation

# %%
# Let's peek at the top 5 Mechanical Score performances in 2024
top_mechanical = df_2024.sort_values(by='Mechanical_Score', ascending=False).head(5)
print("--- TOP 5 MECHANICAL PERFORMANCES IN 2024 (SINGLE MAP) ---")
for idx, row in top_mechanical.iterrows():
    print(f"{row['Player']} ({row['Team']}): {row['Mechanical_Score']:.2f}")

top_clutch = df_2024.sort_values(by='Clutch_Score', ascending=False).head(5)
print("\n--- TOP 5 CLUTCH PERFORMANCES IN 2024 (SINGLE MAP) ---")
for idx, row in top_clutch.iterrows():
    print(f"{row['Player']} ({row['Team']}): {row['Clutch_Score']} (1v1: {row['1v1']}, 1v2: {row['1v2']})")
    
top_entry = df_2024.sort_values(by='Entry_Success', ascending=False).head(5)
print("\n--- TOP 5 ENTRY SUCCESS IN 2024 (SINGLE MAP) ---")
for idx, row in top_entry.iterrows():
    print(f"{row['Player']} ({row['Team']}): {row['Entry_Success']:.2f} (FK: {row['First Kills']}, FD: {row['First Deaths']})")
