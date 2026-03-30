import os
import pandas as pd

class VCTResolver:
    """
    Responsible for unifying Team Names and Player Names into unique Integer IDs.
    Exotic -> distinct 2021 and 2022 IDs.
    TP -> distinct Typhoon and Typhone IDs.
    """
    def __init__(self, data_dir: str = "data/raw/all_ids"):
        self.data_dir = data_dir
        self.team_mappings = None
        self.player_mappings = None
        
    def load_mappings(self):
        """Loads the global ID dictionaries for reference."""
        try:
            self.team_mappings = pd.read_csv(
                os.path.join(self.data_dir, "all_teams_mapping.csv"),
                keep_default_na=False
            )
            self.player_mappings = pd.read_csv(
                os.path.join(self.data_dir, "all_players_ids.csv"),
                keep_default_na=False
            )
        except Exception as e:
            print(f"[Resolver] Error loading ID mappings: {e}")

    def resolve_team_id(self, team_name: str, year: str) -> str:
        """
        Looks up a team's ID securely using the team name and year.
        If a team is labeled 'TBD', return None.
        """
        if team_name == "TBD" or team_name == "":
            return None
            
        if self.team_mappings is None:
            return None

        # This depends on the exact schema of `all_teams_mapping.csv`.
        # Refuse to return placeholder IDs (silent join corruption risk).
        raise NotImplementedError(
            "Team ID resolution is not implemented. "
            "Implement lookup against all_teams_mapping.csv schema."
        )

    def resolve_player_id(self, player_name: str) -> str:
        """Looks up a player's ID securely."""
        if player_name == "nan" or player_name == "":
            return None
        # This depends on the exact schema of `all_players_ids.csv`.
        # Refuse to return placeholder IDs (silent join corruption risk).
        raise NotImplementedError(
            "Player ID resolution is not implemented. "
            "Implement lookup against all_players_ids.csv schema."
        )

if __name__ == "__main__":
    resolver = VCTResolver()
    resolver.load_mappings()
    print("Mappings loaded.")
