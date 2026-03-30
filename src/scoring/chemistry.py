import pandas as pd
import numpy as np
from itertools import combinations


class ChemistryEngine:
    """
    Computes Team Chemistry Score based on how long specific player
    combinations have played together in official VCT matches.

    Logarithmic curve (from ReadMe.md spec):
      0  - 5  matches together => 60% chemistry
      5  - 15 matches together => 80% chemistry
      15+    matches together  => 95 - 100% chemistry
    Absolute floor: 50% for any new pairing (all are T1 pros).
    """

    # Chemistry curve breakpoints
    FLOOR        = 0.60   # complete strangers (raised from 0.50)
    LOW_CAP      = 0.65   # 1-5 matches (raised from 0.60)
    MID_CAP      = 0.85   # 5-15 matches (raised from 0.80)
    HIGH_CAP     = 1.00   # 15+ matches

    def __init__(self, overview_df: pd.DataFrame):
        """
        overview_df must have columns: Match Name, Player, Team, Year.
        This is the full cleaned overview across all 5 years.
        """
        self._pair_history: dict[frozenset, int] = {}
        self._build_pair_history(overview_df)

    # ------------------------------------------------------------------
    # Internal: build pairwise co-match counts from history
    # ------------------------------------------------------------------
    def _build_pair_history(self, overview_df: pd.DataFrame) -> None:
        """
        For every match × team combination, record all pairs of players
        and increment their shared-match count.
        """
        # We only need one row per player per match (collapse across maps)
        roster_per_match = (
            overview_df[['Match Name', 'Team', 'Player', 'Year']]
            .drop_duplicates()
        )

        for (match_name, team), group in roster_per_match.groupby(['Match Name', 'Team']):
            players = group['Player'].tolist()
            for pair in combinations(players, 2):
                key = frozenset(pair)
                self._pair_history[key] = self._pair_history.get(key, 0) + 1

        print(f"[Chemistry] Built history for {len(self._pair_history):,} unique player pairs.")

    # ------------------------------------------------------------------
    # Chemistry conversion: match count -> chemistry multiplier
    # ------------------------------------------------------------------
    def _matches_to_chemistry(self, match_count: int) -> float:
        """
        Converts number of matches played together into a chemistry multiplier:
          0       => 50% floor
          1-5     => scales from 60% to 60% (flat cap)
          5-15    => scales from 60% to 80% (logarithmic growth)
          15+     => scales from 80% to 100% (logarithmic, asymptote)
        """
        if match_count == 0:
            return self.FLOOR

        if match_count <= 5:
            return self.LOW_CAP

        if match_count <= 15:
            # linear interpolation 60% -> 80% over 5-15 matches
            progress = (match_count - 5) / 10  # 0.0 -> 1.0
            return self.LOW_CAP + progress * (self.MID_CAP - self.LOW_CAP)

        # 15+ matches: logarithmic approach toward 100%
        # log growth: 80% at 15 matches, ~95% at 50 matches, ~98% at 100 matches
        log_factor = np.log(match_count - 14) / np.log(100)  # normalised log scale
        return min(self.MID_CAP + log_factor * (self.HIGH_CAP - self.MID_CAP), self.HIGH_CAP)

    # ------------------------------------------------------------------
    # Public: get chemistry score for a roster
    # ------------------------------------------------------------------
    def get_roster_chemistry(self, players: list[str]) -> dict:
        """
        Given a list of 5 player names, return:
          {
            'chemistry_score': float,      # 0.50 - 1.00
            'pair_details': [              # per-pair breakdown
                {'pair': (A, B), 'matches': int, 'chemistry': float},
                ...
            ]
          }
        """
        if len(players) < 2:
            return {'chemistry_score': self.FLOOR, 'pair_details': []}

        pair_details = []
        for pair in combinations(players, 2):
            key     = frozenset(pair)
            matches = self._pair_history.get(key, 0)
            chem    = self._matches_to_chemistry(matches)
            pair_details.append({
                'pair': pair,
                'matches_together': matches,
                'chemistry': round(chem, 4)
            })

        avg_chemistry = np.mean([p['chemistry'] for p in pair_details])

        return {
            'chemistry_score': round(float(avg_chemistry), 4),
            'pair_details': pair_details
        }

    def get_team_chemistry_score(self, players: list[str]) -> float:
        """Convenience method — returns just the float multiplier."""
        return self.get_roster_chemistry(players)['chemistry_score']


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os, warnings
    warnings.filterwarnings('ignore')
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.data.loader  import VCTDataLoader
    from src.data.cleaner import VCTCleaner

    loader   = VCTDataLoader(data_dir='data/raw')
    overview = loader.load_overviews()
    cleaner  = VCTCleaner()
    clean    = cleaner.clean_overviews(overview)

    engine = ChemistryEngine(clean)

    # Test with Gen.G 2024 roster (known long-running lineup)
    geng_2024 = ['t3xture', 'Meteor', 'Lakia', 'Karon', 'Munchkin']
    result = engine.get_roster_chemistry(geng_2024)
    print(f"\nGen.G 2024 Chemistry: {result['chemistry_score']*100:.1f}%")
    for p in sorted(result['pair_details'], key=lambda x: -x['matches_together'])[:5]:
        print(f"  {p['pair'][0]} + {p['pair'][1]}: {p['matches_together']} matches -> {p['chemistry']*100:.0f}%")

    # Compare with a hypothetical new roster
    new_roster = ['aspas', 'TenZ', 'Derke', 'Leo', 'Boaster']
    result2 = engine.get_roster_chemistry(new_roster)
    print(f"\nNew Superteam Chemistry: {result2['chemistry_score']*100:.1f}%")
