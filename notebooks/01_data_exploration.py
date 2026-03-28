import pandas as pd
import os

def explore_2024_data():
    overview_path = os.path.join("data", "raw", "vct_2024", "matches", "overview.csv")
    
    print("--- 1. Testing Default Pandas Loading (The 'nan' bug) ---")
    # Load without any special arguments
    df_default = pd.read_csv(overview_path)
    
    # Let's see if pandas read the player "nan" as a literal NaN (null) value
    null_players = df_default[df_default['Player'].isna()]
    print(f"Total rows where 'Player' column is null/NaN: {len(null_players)}")
    
    print("\n--- 2. Fixing the 'nan' bug ---")
    # Load with keep_default_na=False to prevent "nan" from becoming null
    df_fixed = pd.read_csv(overview_path, keep_default_na=False)
    # Re-replace empty strings with actual nulls if needed, except for 'Player'
    # For now, let's just see how many empty players we have
    empty_players = df_fixed[df_fixed['Player'] == ""]
    nan_string_players = df_fixed[df_fixed['Player'] == "nan"]
    
    print(f"Total rows where 'Player' column is exactly the string 'nan': {len(nan_string_players)}")
    print(f"Total rows where 'Player' column is truly an empty string: {len(empty_players)}")

    print("\n--- 3. Checking for completely missing stats (Chinese events) ---")
    # Check rows where the critical stats like Rating, ACS, Kills, Deaths are empty/missing
    # In df_fixed, missing values are empty strings "" because keep_default_na=False
    missing_stats_mask = (df_fixed['Rating'] == "") & (df_fixed['Average Combat Score'] == "")
    missing_matches = df_fixed[missing_stats_mask]
    
    unique_missing_matches = missing_matches['Match Name'].nunique()
    print(f"Total rows with completely missing basic stats (ACS/Rating): {len(missing_matches)}")
    print(f"Total unique match names affected by missing stats: {unique_missing_matches}")

if __name__ == "__main__":
    explore_2024_data()
