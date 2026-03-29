import pandas as pd


# ------------------------------------------------------------------
# Agent role ontology (VCT 2021-2025 + current meta)
# ------------------------------------------------------------------
AGENT_ROLE_MAP = {
    'duelist': {
        'jett', 'reyna', 'raze', 'phoenix', 'neon', 'yoru', 'soul', 'fade', 'killjoy'  # keep mostly duelist
    },
    'initiator': {
        'sova', 'breach', 'skye', 'kayo', 'breathe', 'fade', 'xaeco'  # optional placeholders
    },
    'controller': {
        'omen', 'viper', 'brimstone', 'astra', 'harbor', 'hannibal'  # roles may vary, static mapping
    },
    'sentinel': {
        'sage', 'killjoy', 'chamber', 'cypher', 'deadlock', 'peror'  # includes Sentinel-like arms
    },
    'flex': {
        'kayo', 'harbor', 'sova', 'omen', 'viper', 'astra', 'fade', 'neon', 'yoru'  # all-rounders
    }
}

# Role champions (highly optimal picks used for extra bonus)
ROLE_STAR_AGENTS = {
    'duelist': {'jett', 'raze', 'neon', 'reyna'},
    'initiator': {'sova', 'breach', 'skye'},
    'controller': {'omen', 'viper', 'brimstone'},
    'sentinel': {'sage', 'killjoy', 'chamber'},
    'flex': {'kayo', 'astra', 'harbor'}
}

REQUIRED_ROLES = {'duelist', 'initiator', 'controller', 'sentinel', 'flex'}


class RoleBalanceEngine:
    """Computes a role coverage/fit score for teams based on roster agents."""

    def __init__(self):
        pass

    @staticmethod
    def normalize_agent_name(agent_name: str) -> str:
        if not isinstance(agent_name, str):
            return ''
        return agent_name.strip().lower().replace(' ', '')

    def assign_role(self, agent_name: str) -> str:
        """Return canonical role for an agent, or 'flex' for unknown/ambiguous agents."""
        agent_key = self.normalize_agent_name(agent_name)
        if not agent_key:
            return 'flex'

        # exact match from predefined map
        for role, agents in AGENT_ROLE_MAP.items():
            if agent_key in agents:
                return role

        return 'flex'

    def compute_team_role_balance(self, agents: list) -> float:
        """Compute team role balance score 0-100 by role coverage + star agent bonuses."""
        if not agents:
            return 0.0

        roles = [self.assign_role(a) for a in agents if isinstance(a, str)]
        unique_roles = set(roles)

        # role coverage component
        covered = unique_roles.intersection(REQUIRED_ROLES)
        missing = REQUIRED_ROLES - covered

        coverage_score = len(covered) * 15  # 0..75
        if not missing:
            coverage_score += 20  # full role synergy bonus

        # penalty for missing roles
        missing_penalty = len(missing) * 10

        # star agent bonus
        star_bonus = 0
        for role, star_agents in ROLE_STAR_AGENTS.items():
            for agent in agents:
                if self.assign_role(agent) == role and self.normalize_agent_name(agent) in star_agents:
                    star_bonus += 3

        # cap and normalize
        raw = coverage_score + star_bonus - missing_penalty
        role_balance = max(0.0, min(raw, 100.0))
        return round(role_balance, 2)

    def team_role_balance_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a DataFrame with team role_balance per match team."""
        if 'Match Name' not in df.columns or 'Team' not in df.columns or 'Agents' not in df.columns:
            raise ValueError("DataFrame must include 'Match Name', 'Team', and 'Agents' columns")

        # `Agents` may be a list or semicolon-separated string.
        def parse_agents(cell):
            if isinstance(cell, list):
                return [str(a) for a in cell if pd.notna(a)]
            if isinstance(cell, str):
                sep = ';' if ';' in cell else ','
                return [x.strip() for x in cell.split(sep) if x.strip()]
            return []

        grouped = []
        for (match_name, team), sub in df.groupby(['Match Name', 'Team']):
            agent_commit = []
            for cell in sub['Agents'].fillna(''):
                agent_commit.extend(parse_agents(cell))

            # unique picks per team; we expect 5 unique values but can have duplicates in raw data
            agent_list = [a for a in dict.fromkeys(agent_commit) if a]
            if len(agent_list) > 5:
                agent_list = agent_list[:5]

            balance = self.compute_team_role_balance(agent_list)
            grouped.append({'Match Name': match_name, 'Team': team, 'Role_Balance_Score': balance})

        return pd.DataFrame(grouped)


if __name__ == '__main__':
    engine = RoleBalanceEngine()
    sample = pd.DataFrame([
        {'Match Name': 'm1', 'Team': 'A', 'Agents': 'Jett;Omen;Sova;Killjoy;Sage'},
        {'Match Name': 'm1', 'Team': 'B', 'Agents': 'Raze;Viper;Breach;Cypher;Skye'},
    ])
    print(engine.team_role_balance_from_df(sample))
