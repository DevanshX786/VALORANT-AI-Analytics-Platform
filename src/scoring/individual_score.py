import pandas as pd
import numpy as np


class PlayerScorer:
    """
    Computes the 7-pillar individual player scores based on raw VCT match data.
    """

    CONSISTENCY_WINDOW = 15  # matches to look back for consistency

    def __init__(self):
        pass

    def calculate_mechanical_score(self, df: pd.DataFrame) -> pd.Series:
        """ACS, K/D, and Multi-kills -> raw fragging power."""
        acs    = pd.to_numeric(df.get('Average Combat Score'), errors='coerce').fillna(0)
        kills  = pd.to_numeric(df.get('Kills'),  errors='coerce').fillna(0)
        deaths = pd.to_numeric(df.get('Deaths'), errors='coerce').replace(0, 1).fillna(1)
        k2 = pd.to_numeric(df.get('2k'), errors='coerce').fillna(0)
        k3 = pd.to_numeric(df.get('3k'), errors='coerce').fillna(0)
        k4 = pd.to_numeric(df.get('4k'), errors='coerce').fillna(0)
        k5 = pd.to_numeric(df.get('5k'), errors='coerce').fillna(0)
        return (acs * 0.45) + ((kills / deaths) * 30) + ((k2*2 + k3*3 + k4*4 + k5*5) * 2)

    def calculate_clutch_score(self, df: pd.DataFrame) -> pd.Series:
        """Exponentially weighted 1vX clutch scenarios."""
        c1 = pd.to_numeric(df.get('1v1'), errors='coerce').fillna(0) * 1
        c2 = pd.to_numeric(df.get('1v2'), errors='coerce').fillna(0) * 2
        c3 = pd.to_numeric(df.get('1v3'), errors='coerce').fillna(0) * 4
        c4 = pd.to_numeric(df.get('1v4'), errors='coerce').fillna(0) * 8
        c5 = pd.to_numeric(df.get('1v5'), errors='coerce').fillna(0) * 15
        return c1 + c2 + c3 + c4 + c5

    def calculate_entry_success(self, df: pd.DataFrame) -> pd.Series:
        """First Kills / (FK + FD) ratio + FKD differential."""
        fk  = pd.to_numeric(df.get('First Kills'),        errors='coerce').fillna(0)
        fd  = pd.to_numeric(df.get('First Deaths'),       errors='coerce').fillna(0)
        fkd = pd.to_numeric(df.get('Kills - Deaths (FKD)'), errors='coerce').fillna(0)
        fk_ratio = fk / (fk + fd).replace(0, 1)
        return (fk_ratio * 100) + fkd

    def calculate_utility_score(self, df: pd.DataFrame) -> pd.Series:
        """Assists + Spike Plants + Spike Defuses."""
        assists  = pd.to_numeric(df.get('Assists'),        errors='coerce').fillna(0)
        plants   = pd.to_numeric(df.get('Spike Plants'),   errors='coerce').fillna(0)
        defuses  = pd.to_numeric(df.get('Spike Defuses'),  errors='coerce').fillna(0)
        return assists + (plants * 0.5) + (defuses * 0.5)

    def calculate_economic_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """Econ rating: damage dealt per 1000 credits spent."""
        econ = pd.to_numeric(df.get('Econ'), errors='coerce').fillna(1.0)
        return econ * 10

    def calculate_consistency_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Reward stable performers, punish coin-flip players.

        Logic:
          - Group all rows by Player (across all matches/maps).
          - Sort each player's history chronologically by Year.
          - Compute rolling std of Rating over the last CONSISTENCY_WINDOW matches.
          - Consistency = 1 / (rolling_std + 0.1)   (higher = more consistent)
          - Players with <3 data points get a neutral score of 5.0.
          - Merge back onto the original df index.
        """
        rating_col = 'Rating'
        player_col = 'Player'
        year_col   = 'Year'

        if rating_col not in df.columns:
            return pd.Series(5.0, index=df.index)

        temp = df[[player_col, year_col, rating_col]].copy()
        temp[rating_col] = pd.to_numeric(temp[rating_col], errors='coerce')

        # sort chronologically so rolling window respects time order
        temp = temp.sort_values([player_col, year_col])

        # rolling std per player
        temp['_roll_std'] = (
            temp.groupby(player_col)[rating_col]
                .transform(lambda x: x.rolling(self.CONSISTENCY_WINDOW, min_periods=3).std())
        )

        # 1 / (std + 0.1) — scale by 10 to bring into same ballpark as other scores
        temp['_consistency'] = (1 / (temp['_roll_std'].fillna(0.5) + 0.1)) * 10

        # Reindex back to original df order
        return temp['_consistency'].reindex(df.index).fillna(5.0)

    # ------------------------------------------------------------------
    # Map Specialization — computed outside of per-row context
    # ------------------------------------------------------------------
    def build_player_map_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by (Player, Map) with columns:
          map_win_rate, map_acs_mean, map_kd_mean
        Used by prediction_engine to add map-context to team strength.
        """
        tmp = df.copy()
        tmp['_acs']  = pd.to_numeric(tmp.get('Average Combat Score'), errors='coerce')
        tmp['_kd']   = (
            pd.to_numeric(tmp.get('Kills'), errors='coerce').fillna(0) /
            pd.to_numeric(tmp.get('Deaths'), errors='coerce').replace(0, 1).fillna(1)
        )
        # 'Side' column tells us Attack/Defense; we approximate 'won this map'
        # from the scores.csv join (handled in prediction_engine, not here).
        map_stats = (
            tmp.groupby(['Player', 'Map'])
               .agg(map_acs_mean=('_acs', 'mean'),
                    map_kd_mean=('_kd',  'mean'),
                    map_appearances=('_acs', 'count'))
               .reset_index()
        )
        return map_stats

    # ------------------------------------------------------------------
    # Master scorer
    # ------------------------------------------------------------------
    def compute_overall_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all sub-scores and append them as columns."""
        out = df.copy()
        out['Mechanical_Score']  = self.calculate_mechanical_score(out)
        out['Clutch_Score']      = self.calculate_clutch_score(out)
        out['Entry_Success']     = self.calculate_entry_success(out)
        out['Utility_Score']     = self.calculate_utility_score(out)
        out['Economic_Score']    = self.calculate_economic_efficiency(out)
        out['Consistency_Score'] = self.calculate_consistency_score(out)
        return out


if __name__ == "__main__":
    scorer = PlayerScorer()
    print("PlayerScorer loaded — 6 pillars active.")
