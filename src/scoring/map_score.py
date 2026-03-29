import pandas as pd
import numpy as np
import warnings
import os
import sys

warnings.filterwarnings('ignore')


class MapScoreEngine:
    """
    Builds per-player, per-map performance profiles using:
      - overview.csv  : player stats on each map (ACS, K/D, Kills, Deaths)
      - maps_scores.csv: team-level attacker/defender side win counts per map

    Outputs a lookup table indexed by (Player, Map) with:
      map_acs_mean     -> Average Combat Score on this map
      map_kd_mean      -> Kill/Death ratio on this map
      map_appearances  -> How many times played (reliability weight)
      map_win_rate     -> Win rate on this map (from maps_scores join)
      atk_win_rate     -> Normalised attacker-side win rate
      def_win_rate     -> Normalised defender-side win rate
      map_score        -> Composite score for prediction weighting
    """

    # Global VCT side-bias baselines (approximated from dataset)
    # Attack wins ~50-52% of rounds across most maps
    GLOBAL_ATK_BASELINE = 0.51
    GLOBAL_DEF_BASELINE = 0.49

    def __init__(self):
        self._player_map_table: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Build the player-map lookup table
    # ------------------------------------------------------------------
    def build(self, overview_df: pd.DataFrame, maps_scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        overview_df    : cleaned overview across all years (has Player, Map, Team, ACS, K, D etc.)
        maps_scores_df : maps_scores across all years (has Match Name, Map, Team A/B, scores)

        Returns a DataFrame indexed by [Player, Map] for fast lookup.
        """
        # ---- Part A: player-level map stats from overview ----
        ov = overview_df.copy()
        ov['_acs']    = pd.to_numeric(ov.get('Average Combat Score'), errors='coerce')
        ov['_kills']  = pd.to_numeric(ov.get('Kills'),  errors='coerce').fillna(0)
        ov['_deaths'] = pd.to_numeric(ov.get('Deaths'), errors='coerce').replace(0, 1).fillna(1)
        ov['_kd']     = ov['_kills'] / ov['_deaths']

        player_map_stats = (
            ov.groupby(['Player', 'Map'])
              .agg(
                  map_acs_mean    = ('_acs',  'mean'),
                  map_kd_mean     = ('_kd',   'mean'),
                  map_appearances = ('_acs',  'count'),
              )
              .reset_index()
        )

        # ---- Part B: team win rates from maps_scores ----
        ms = maps_scores_df.copy()

        # Convert side scores to numeric
        for col in ['Team A Score', 'Team A Attacker Score', 'Team A Defender Score',
                    'Team B Score', 'Team B Attacker Score', 'Team B Defender Score']:
            ms[col] = pd.to_numeric(ms.get(col), errors='coerce').fillna(0)

        # Build team-level map records (one row per team per map appearance)
        team_records = []
        for _, row in ms.iterrows():
            map_name = row['Map']
            # Team A
            a_total = row['Team A Score'] + row['Team B Score']
            if a_total > 0:
                team_records.append({
                    'Team':    row['Team A'],
                    'Map':     map_name,
                    'won':     int(row['Team A Score'] > row['Team B Score']),
                    'atk_rds': row['Team A Attacker Score'],
                    'def_rds': row['Team A Defender Score'],
                    'total_rds_played': row['Team A Score'] + row['Team B Score'],
                })
                # Team B
                team_records.append({
                    'Team':    row['Team B'],
                    'Map':     map_name,
                    'won':     int(row['Team B Score'] > row['Team A Score']),
                    'atk_rds': row['Team B Attacker Score'],
                    'def_rds': row['Team B Defender Score'],
                    'total_rds_played': row['Team A Score'] + row['Team B Score'],
                })

        team_df = pd.DataFrame(team_records)

        team_map_stats = (
            team_df.groupby(['Team', 'Map'])
                   .agg(
                       team_win_rate = ('won',      'mean'),
                       total_atk_rds = ('atk_rds',  'sum'),
                       total_def_rds = ('def_rds',  'sum'),
                       total_rds     = ('total_rds_played', 'sum'),
                   )
                   .reset_index()
        )
        team_map_stats['atk_win_rate'] = (
            team_map_stats['total_atk_rds'] /
            (team_map_stats['total_rds'] / 2).replace(0, 1)
        ).clip(0, 1)
        team_map_stats['def_win_rate'] = (
            team_map_stats['total_def_rds'] /
            (team_map_stats['total_rds'] / 2).replace(0, 1)
        ).clip(0, 1)

        # ---- Part C: join player stats with their historical team win rates ----
        # We need to know which team a player played for per map.
        # Use overview to get (Player, Map, Team) combos, then merge team stats.
        player_team_map = (
            ov[['Player', 'Map', 'Team']]
            .drop_duplicates()
        )
        player_with_team_stats = pd.merge(
            player_team_map,
            team_map_stats[['Team', 'Map', 'team_win_rate', 'atk_win_rate', 'def_win_rate']],
            on=['Team', 'Map'],
            how='left'
        )

        # Average across all teams the player has been on for each map
        player_map_win = (
            player_with_team_stats
            .groupby(['Player', 'Map'])
            .agg(
                map_win_rate  = ('team_win_rate', 'mean'),
                atk_win_rate  = ('atk_win_rate',  'mean'),
                def_win_rate  = ('def_win_rate',  'mean'),
            )
            .reset_index()
        )

        # ---- Part D: merge everything together ----
        result = pd.merge(player_map_stats, player_map_win, on=['Player', 'Map'], how='left')

        # Normalise side rates against global baseline
        result['atk_advantage'] = result['atk_win_rate'].fillna(self.GLOBAL_ATK_BASELINE) - self.GLOBAL_ATK_BASELINE
        result['def_advantage'] = result['def_win_rate'].fillna(self.GLOBAL_DEF_BASELINE) - self.GLOBAL_DEF_BASELINE

        # ---- Part E: composite map_score ----
        # Blend mechanical map performance + win rate contribution
        # Reliability weight: more appearances = more trustworthy score
        reliability = (result['map_appearances'].clip(1, 20) / 20)  # caps at 1.0 after 20+ maps

        acs_norm    = result['map_acs_mean'].fillna(0)   / 300        # typical T1 ACS ~200-280
        kd_norm     = (result['map_kd_mean'].fillna(1)   - 1) * 0.5  # KD centred, scaled
        win_contrib = result['map_win_rate'].fillna(0.5) - 0.5        # above/below 50% win rate

        result['map_score'] = (
            (acs_norm * 0.4 + kd_norm * 0.35 + win_contrib * 0.25) * reliability
        ) * 100   # scale to readable range

        self._player_map_table = result.set_index(['Player', 'Map'])
        print(f"[MapScore] Built profiles for {len(result['Player'].unique()):,} players across {len(result['Map'].unique())} maps.")
        return self._player_map_table

    # ------------------------------------------------------------------
    # Public: look up a player's score on a specific map
    # ------------------------------------------------------------------
    def get_player_map_score(self, player: str, map_name: str) -> dict:
        """Returns map performance data for a specific player on a specific map."""
        map_name_clean = map_name.strip().title()
        try:
            row = self._player_map_table.loc[(player, map_name_clean)]
            return {
                'map_score':     round(float(row.get('map_score', 0)),    3),
                'map_acs':       round(float(row.get('map_acs_mean', 0)), 1),
                'map_kd':        round(float(row.get('map_kd_mean', 1)),  2),
                'win_rate':      round(float(row.get('map_win_rate', 0.5)), 3),
                'atk_advantage': round(float(row.get('atk_advantage', 0)), 3),
                'def_advantage': round(float(row.get('def_advantage', 0)), 3),
                'appearances':   int(row.get('map_appearances', 0)),
            }
        except KeyError:
            # Player never appeared on this map — return neutral defaults
            return {
                'map_score': 0.0, 'map_acs': 0.0, 'map_kd': 1.0,
                'win_rate': 0.5,  'atk_advantage': 0.0, 'def_advantage': 0.0,
                'appearances': 0,
            }

    def get_team_map_score(self, players: list, map_name: str) -> float:
        """Average map_score across all 5 players for a given map."""
        scores = [self.get_player_map_score(p, map_name)['map_score'] for p in players]
        return round(float(np.mean(scores)), 3)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.data.loader  import VCTDataLoader
    from src.data.cleaner import VCTCleaner

    loader = VCTDataLoader(data_dir='data/raw')
    print("Loading overview...")
    overview    = loader.load_overviews()
    maps_scores = loader.load_maps_scores()

    cleaner  = VCTCleaner()
    clean_ov = cleaner.clean_overviews(overview)

    engine = MapScoreEngine()
    engine.build(clean_ov, maps_scores)

    # Test: aspas performance on Ascent vs Split
    for map_name in ['Ascent', 'Split', 'Bind', 'Haven']:
        data = engine.get_player_map_score('aspas', map_name)
        print(f"  aspas on {map_name:<10}: map_score={data['map_score']:>6.2f}  acs={data['map_acs']:.0f}  win_rate={data['win_rate']*100:.0f}%  appearances={data['appearances']}")

    # Team comparison on Ascent
    geng  = ['t3xture', 'Meteor', 'Lakia', 'Karon', 'Munchkin']
    sents = ['Sacy', 'TenZ', 'Zellsis', 'johnqt', 'zekken']
    print(f"\nGen.G on Ascent:    {engine.get_team_map_score(geng, 'Ascent'):.3f}")
    print(f"Sentinels on Ascent: {engine.get_team_map_score(sents, 'Ascent'):.3f}")
