from typing import Dict


class PredictionEngine:
    """Applies format and stage modifiers to team-level features."""

    FORMAT_WEIGHTS = {
        'Bo1': {'map_score': 1.2, 'chemistry': 0.8, 'consistency': 0.85},
        'Bo3': {'map_score': 1.0, 'chemistry': 1.0, 'consistency': 1.0},
        'Bo5': {'map_score': 0.8, 'chemistry': 1.2, 'consistency': 1.1},
    }

    STAGE_MULTIPLIER = {
        'Group': 1.0,
        'Playoffs': 1.3,
        'Playoff': 1.3,
    }

    def __init__(self, match_format: str = 'Bo3', stage: str = 'Group'):
        self.match_format = match_format if match_format in self.FORMAT_WEIGHTS else 'Bo3'
        self.stage = stage if stage in self.STAGE_MULTIPLIER else 'Group'

    def normalize_format(self, fmt: str) -> str:
        fmt = str(fmt).strip().upper()
        return 'Bo1' if fmt == 'BO1' else 'Bo3' if fmt == 'BO3' else 'Bo5' if fmt == 'BO5' else 'Bo3'

    def normalize_stage(self, stage: str) -> str:
        stage = str(stage).strip().capitalize()
        if stage in ['Group', 'Groups', 'Group Stage']:
            return 'Group'
        if stage in ['Playoffs', 'Playoff']:
            return 'Playoffs'
        return 'Group'

    def get_format_weights(self, fmt: str = None) -> Dict[str, float]:
        fmt = self.normalize_format(fmt or self.match_format)
        return self.FORMAT_WEIGHTS.get(fmt, self.FORMAT_WEIGHTS['Bo3'])

    def get_stage_multiplier(self, stage: str = None) -> float:
        stage = self.normalize_stage(stage or self.stage)
        return self.STAGE_MULTIPLIER.get(stage, 1.0)

    def apply_modifiers(self, team_features: Dict[str, float], fmt: str = None, stage: str = None) -> Dict[str, float]:
        fmt_weights = self.get_format_weights(fmt)
        stage_mul = self.get_stage_multiplier(stage)

        adjusted = team_features.copy()

        for key in ['map_score', 'chemistry', 'consistency']:
            if key in adjusted:
                adjusted[key] = adjusted[key] * fmt_weights.get(key, 1.0)

        # Apply stage pressure to clutch and chemistry
        if 'clutch_sum' in adjusted:
            adjusted['clutch_sum'] = adjusted['clutch_sum'] * stage_mul
        if 'chemistry' in adjusted:
            adjusted['chemistry'] = adjusted['chemistry'] * stage_mul

        return adjusted
