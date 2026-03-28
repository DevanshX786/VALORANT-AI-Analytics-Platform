import os
import pandas as pd
from typing import Dict, List

class VCTDataLoader:
    """
    Responsible for loading raw Kaggle CSV files from the 2021-2025 structure
    and concatenating them into unified DataFrames.
    """
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.years = ["vct_2021", "vct_2022", "vct_2023", "vct_2024", "vct_2025"]
    
    def _read_csv_safe(self, path: str) -> pd.DataFrame:
        """Reads a CSV safely, avoiding the 'nan' player parsing bug."""
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=False)

    def load_overviews(self) -> pd.DataFrame:
        """Loads and concatenates matches/overview.csv from all years."""
        dfs = []
        for year in self.years:
            path = os.path.join(self.data_dir, year, "matches", "overview.csv")
            df = self._read_csv_safe(path)
            if not df.empty:
                df['Year'] = year[-4:] # Add the year for tracking
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_scores(self) -> pd.DataFrame:
        """Loads and concatenates matches/scores.csv from all years."""
        dfs = []
        for year in self.years:
            path = os.path.join(self.data_dir, year, "matches", "scores.csv")
            df = self._read_csv_safe(path)
            if not df.empty:
                df['Year'] = year[-4:]
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_team_mappings(self) -> pd.DataFrame:
        """Loads and concatenates matches/team_mapping.csv from all years."""
        dfs = []
        for year in self.years:
            path = os.path.join(self.data_dir, year, "matches", "team_mapping.csv")
            df = self._read_csv_safe(path)
            if not df.empty:
                df['Year'] = year[-4:]
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_kills_stats(self) -> pd.DataFrame:
        """Loads and concatenates matches/kills_stats.csv from all years."""
        dfs = []
        for year in self.years:
            path = os.path.join(self.data_dir, year, "matches", "kills_stats.csv")
            df = self._read_csv_safe(path)
            if not df.empty:
                df['Year'] = year[-4:]
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    loader = VCTDataLoader()
    print("Testing loader...")
    try:
        overview_df = loader.load_overviews()
        print(f"Successfully loaded {len(overview_df)} total overview records across all years.")
    except Exception as e:
        print(f"Error loading data: {e}")
