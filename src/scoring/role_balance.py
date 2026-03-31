import pandas as pd


# ------------------------------------------------------------------
# Agent role ontology (VCT current meta, 30 agents)
# ------------------------------------------------------------------
FULL_AGENT_POOL = {
    'brimstone','viper','omen','astra','harbor','clove','miks',
    'phoenix','jett','reyna','raze','yoru','neon','iso','waylay',
    'sova','breach','skye','kayo','fade','gekko','tejo',
    'killjoy','cypher','sage','chamber','deadlock','vyse','veto'
}

AGENT_ROLE_MAP = {
    'duelist': {'phoenix', 'jett', 'reyna', 'raze', 'yoru', 'neon', 'iso', 'waylay'},
    'initiator': {'sova', 'breach', 'skye', 'kayo', 'fade', 'gekko', 'tejo'},
    'controller': {'brimstone', 'viper', 'omen', 'astra', 'harbor', 'clove', 'miks'},
    'sentinel': {'killjoy', 'cypher', 'sage', 'chamber', 'deadlock', 'vyse', 'veto'},
    'flex': {'neon', 'yoru', 'kayo', 'fade', 'gekko', 'tejo', 'harbor', 'miks', 'clove', 'deadlock', 'vyse', 'veto'}
}

# map known variants to canonical names (including spelling variants from raw data)
AGENT_CANONICAL_ALIAS = {
    # Accept British spelling / common typos, canonicalize to Riot spelling.
    'harbour': 'harbor',
    'harbore': 'harbor',
    'soul': 'iso',
    'hbir': 'harbor',
    'ovrd': 'veto',
    'kaio': 'kayo',
}

REQUIRED_ROLES = {'duelist', 'initiator', 'controller', 'sentinel'}


class RoleBalanceEngine:
    """Computes a role coverage/fit score for teams based on roster agents."""

    _known_unknown_agents = set()

    def __init__(self):
        pass

    @staticmethod
    def normalize_agent_name(agent_name: str) -> str:
        if not isinstance(agent_name, str):
            return ''
        return agent_name.strip().lower().replace(' ', '').replace('-', '')

    def canonical_agent_name(self, agent_name: str) -> str:
        key = self.normalize_agent_name(agent_name)
        if key in AGENT_CANONICAL_ALIAS:
            return AGENT_CANONICAL_ALIAS[key]
        return key

    def assign_role(self, agent_name: str) -> str:
        """Return canonical role for an agent, or 'flex' for unknown/ambiguous agents."""
        agent_key = self.canonical_agent_name(agent_name)
        if not agent_key:
            return 'flex'

        for role, agents in AGENT_ROLE_MAP.items():
            if agent_key in agents:
                return role

        return 'flex'  # unknown agent treated as flex by default

    def compute_team_role_balance(self, agents: list, comfort_count: int = 0) -> float:
        """
        Compute team role balance score 0-100.
        Logic:
          - Role Coverage: 15 per unique required role (max 60)
          - Full Synergy: +20 if all 4 roles are present
          - Comfort Bonus: +7.5 per comfort agent (max 37.5)
          - Missing Penalty: -10 per missing role
        """
        if not agents:
            return 0.0

        roles = [self.assign_role(a) for a in agents if isinstance(a, str)]
        unique_roles = set(roles)

        # unknown agent namespace diagnostics
        unknown_agents = set()
        for agent in agents:
            canonical = self.canonical_agent_name(agent)
            if canonical and canonical not in FULL_AGENT_POOL:
                unknown_agents.add(agent)

        if unknown_agents:
            new_unknown = unknown_agents - self._known_unknown_agents
            if new_unknown:
                self._known_unknown_agents.update(new_unknown)
                print(f"[RoleBalance] warning: unknown agent names detected: {sorted(new_unknown)}")

        # role coverage component
        covered = unique_roles.intersection(REQUIRED_ROLES)
        missing = REQUIRED_ROLES - covered

        coverage_score = len(covered) * 15  # 0..60
        if not missing:
            coverage_score += 20  # full synergy bonus -> 80

        # penalty for missing roles
        missing_penalty = len(missing) * 10

        # comfort agent bonus (User feedback: replace stars with comfort)
        comfort_bonus = min(comfort_count, 5) * 7.5

        # cap and normalize
        raw = coverage_score + comfort_bonus - missing_penalty
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
