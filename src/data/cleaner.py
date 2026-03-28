import pandas as pd
import numpy as np

class VCTCleaner:
    """
    Cleans raw concatenated data by handling nulls safely
    and applying exclusion logic (e.g. dropping Chinese events without stats)
    """
    def __init__(self):
        # Event columns where a missing value simply means it didn't happen (value = 0)
        self.event_columns = [
            '2k', '3k', '4k', '5k', 
            '1v1', '1v2', '1v3', '1v4', '1v5', 
            'Player Kills', 'Enemy Kills', 'Difference',
            'First Kills', 'First Deaths', 'First Kills - Deaths',
            'First Kills - Deaths (FKD)'
        ]
        
    def clean_overviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the overview.csv dataframe.
        """
        if df.empty:
            return df
            
        print(f"Original overview count: {len(df)}")
        
        # 1. Fill event nulls with 0
        for col in self.event_columns:
            if col in df.columns:
                # Replace empty strings with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        # 2. Identify missing Chinese events (no stats)
        # We assume if 'Rating' and 'Average Combat Score' are entirely empty, the match is incomplete.
        if 'Rating' in df.columns and 'Average Combat Score' in df.columns:
            missing_stats_mask = (df['Rating'] == "") & (df['Average Combat Score'] == "")
            missing_matches = df[missing_stats_mask]['Match Name'].unique()
            if len(missing_matches) > 0:
                print(f"Dropping {len(missing_matches)} matches entirely due to missing basic stats.")
                df = df[~df['Match Name'].isin(missing_matches)]
                
        # 3. Handle TBD teams
        if 'Team' in df.columns:
            tbd_mask = df['Team'] == 'TBD'
            print(f"Dropping {tbd_mask.sum()} player rows belonging to 'TBD' placeholder teams.")
            df = df[~tbd_mask]
        
        print(f"Cleaned overview count: {len(df)}")
        return df

if __name__ == "__main__":
    cleaner = VCTCleaner()
    print("Cleaner module loaded.")
