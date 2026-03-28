================================================================================
  VALORANT AI ANALYTICS PLATFORM
  Match Prediction, Roster Intelligence & Simulation System
================================================================================

OVERVIEW
--------
The VALORANT AI Analytics Platform is a machine learning system designed to
predict match outcomes using dynamic roster-aware modeling.

Unlike traditional esports models that treat teams as fixed units, this system
evaluates individual player performance across multiple seasons and dynamically
recalculates team strength based on current roster composition.

It incorporates:
  - Player transfers and roster changes
  - Team chemistry evolution over time
  - Map-specific and side-specific strengths

to produce realistic, context-aware match predictions.


================================================================================
  CORE PROBLEM & SOLUTION
================================================================================

THE PROBLEM WITH TRADITIONAL SYSTEMS
--------------------------------------
Most esports prediction models assume:  "Team = fixed entity"
But in reality, rosters change constantly.

EXAMPLE SCENARIO
-----------------
  2024:  Team A -> a, b, c, d, e       Team B -> v, w, x, y, z
  2025:  Team A -> a, b, c, d, g       Team B -> t, q, x, y, z
                  (g replaces e)               (t, q replace v, w)

Traditional models still treat Team A as the same team, ignoring new players
and chemistry shifts entirely.

THIS SYSTEM'S SOLUTION
-----------------------
  - Player performance is tracked individually across all teams and seasons
  - New players introduce a chemistry penalty on joining
  - Chemistry improves logarithmically as players play more matches together
  - Final prediction uses updated team strength, not outdated historical data
  - Rookies get a minimum performance floor (tier 1 pros earn their spot)


================================================================================
  SYSTEM ARCHITECTURE
================================================================================

All modules are built on a shared core engine:

  Player Data  ->  Player Scores  ->  Team Strength  ->  Context  ->  Prediction

Context factors:
  - Map performance and side tendencies (attack vs defense)
  - Role balance across the roster
  - Team chemistry score
  - Match format (Bo1 / Bo3 / Bo5)
  - Tournament stage (group stage vs playoffs)


================================================================================
  MODULE 1 - MATCH PREDICTION ENGINE
================================================================================

Goal: Predict match outcomes using current rosters.

INPUT
  - Team A (5 players) and Team B (5 players)
  - Map pool
  - Match format: Bo1 / Bo3 / Bo5
  - Time window: last 6 months / 1 year / 2 years / all time

HOW IT WORKS
  1. Calculate individual player scores for all 10 players
  2. Aggregate into team strength scores
  3. Apply chemistry adjustments, map advantages, and role balance
  4. Apply match format modifier and tournament stage modifier
  5. Feed all features into trained XGBoost model

OUTPUT (Full Breakdown)
  - Win probability % per team
  - Player matchup analysis (who wins their individual lane)
  - Map-by-map odds across the pool
  - Recommended map bans per team
  - Key factors driving the prediction
    e.g. "Team A has +15% chemistry advantage on Ascent"


================================================================================
  MODULE 2 - ROSTER CHANGE IMPACT ANALYSIS
================================================================================

Goal: Evaluate whether a roster change improves or weakens a team.

EXAMPLE
  Original: Team(a, b, c, d, e)
  Updated:  Team(a, b, c, d, x)
  delta   = strength(a,b,c,d,x) - strength(a,b,c,d,e)

OUTPUT
  - Performance change (%)
  - Chemistry impact of the swap
  - Role balance shift
  - Projected team strength trajectory as chemistry builds


================================================================================
  MODULE 3 - PLAYER INTELLIGENCE SYSTEM
================================================================================

3A - PLAYER COMPARISON
  Compare any two players across:
    - Mechanical skill: ACS, K/D, HS%, first blood rate
    - Utility impact: agent-adjusted performance delta
    - Map-specific and side-specific performance
    - Clutch ability and playoff performance

  Two comparison modes:

  Without Abilities (Pure Mechanical Skill)
    Metrics : ACS, K/D, HS%, accuracy
    Answers : "Who is the better pure aimer and game sense player?"

  With Abilities (Agent-Adjusted Performance)
    Metrics : All of the above + utility damage, assists, agent win rate
    Answers : "Who is the better complete VALORANT player?"

  Filters supported: time window, map, opponent

3B - CLUTCH PROBABILITY SYSTEM
  Estimate win probability in 1v1 through 1v5 scenarios.
  Factors in mechanical skill, agent utility, map, and opponent context.


================================================================================
  MODULE 4 - AGENT RECOMMENDATION SYSTEM
================================================================================

Goal: Suggest the optimal 5-agent composition for a team on a given map.

INPUT
  - Your team's 5 players
  - Opponent team (system fetches their most recent agent composition)
  - Map being played

LOGIC
  1. Rank each player's agents by historical performance on this map
  2. Enforce role coverage: Duelist, Initiator, Controller, Sentinel, Flex
  3. Counter-pick opponent's last composition where possible
  4. Never sacrifice player comfort for a counter-pick

OUTPUT
  - Recommended agent per player
  - Role each player fills
  - Reasoning per recommendation
  - Opponent composition it is designed to counter


================================================================================
  MODULE 5 - CUSTOM MATCH SIMULATOR
================================================================================

Goal: Simulate a match between any two custom 5-player teams.

EXAMPLE
  Team A: TenZ, Aspas, Derke, Chronicle, FNS
  Team B: Yay, Something, Leo, Crashies, Boaster

CHEMISTRY HANDLING FOR UNKNOWN COMBINATIONS
  Players who have never played together are fully allowed.
  Unknown combinations start at a 50% chemistry floor.
  Rationale: all players in this dataset are tier 1 professionals.
  Even with zero shared history, individual skill still dominates.

OUTPUT
  - Win probability per team
  - Per-player performance projections
  - Weakest matchup (your team's liability)
  - Strongest matchup (your team's biggest advantage)
  - Recommended agent composition for both teams


================================================================================
  SCORING SYSTEM
================================================================================

INDIVIDUAL PLAYER SCORE
  Built from the player's full match history across all teams and seasons.
  Score travels with the player, not the team.
  Respects the user-selected time window.

  Sub-scores:
    mechanical_score   ACS, K/D, HS%, first blood rate (no agent influence)
    utility_score      Performance delta on utility-heavy agents
    map_scores         Per-map win rate and stat performance
    side_scores        Attack and defense win rate, normalized per map
    clutch_score       Round win rate in 1vX situations
    playoff_score      Performance specifically in elimination matches
    rookie_floor       Applied for players with fewer than 10 tier 1 matches
                       Floor = bottom 25th percentile of T1 players,
                       adjusts upward dynamically as matches increase

TEAM CHEMISTRY SCORE
  Measures how well a specific group of 5 players performs together.

  Decay curve (logarithmic):
    0-5 matches    ->  60% of full chemistry
    5-15 matches   ->  80% of full chemistry
    15+ matches    ->  95-100% of full chemistry

  Minimum floor: 50% chemistry for any combination (tier 1 pros baseline)

ROLE BALANCE SCORE
  Required roles: Duelist / Initiator / Controller / Sentinel / Flex
  Missing role    ->  penalty applied
  Full coverage   ->  bonus applied
  Best-fit role   ->  bonus if player's assigned role matches strongest agent class

MAP PERFORMANCE SCORE
  Per team, per map:
    - Historical win rate
    - Attack win rate  (normalized against global map side bias)
    - Defense win rate (normalized against global map side bias)
  Used for: map odds, recommended bans, Bo3/Bo5 series prediction

MATCH FORMAT MODIFIER
  Bo1  ->  amplify map-specific scores, reduce chemistry weight
  Bo3  ->  balanced weights across all scores
  Bo5  ->  amplify chemistry, side balance, and consistency scores

TOURNAMENT STAGE MODIFIER
  Group stage  ->  standard weights
  Playoffs     ->  playoff_score weighted at 1.3x


================================================================================
  FULL PREDICTION PIPELINE
================================================================================

  User Input: Team A + Team B + Map Pool + Format + Time Window
                              |
                    Individual Player Scores
                              |
                      Team Chemistry Scores
                              |
                      Role Balance Scores
                              |
               Map Performance + Side Tendency Scores
                              |
                    Opponent Context Scores
                              |
               Match Format Modifier (Bo1 / Bo3 / Bo5)
                              |
             Tournament Stage Modifier (Group / Playoff)
                              |
                        XGBoost Model
                              |
       Win % | Player Matchups | Map Odds | Bans | Key Factors


================================================================================
  DATASET
================================================================================

  Source  : Kaggle - VCT Champion Tour 2021-2025
  Path    : D:/archive/
  Size    : ~1.3 GB across 5 year folders

  Primary files (load first, used in Phase 1-3):
    matches/scores.csv                  Match results        (target variable)
    matches/overview.csv                Player stats         (key features)
    matches/maps_played.csv             Maps played per match
    matches/maps_scores.csv             Map-level scores and side data
    matches/team_mapping.csv            Player-to-team links (key for chemistry)
    players_stats/players_stats.csv     Aggregated player stats per season

  Secondary files (Phase 4+):
    agents/agents_pick_rates.csv
    agents/teams_picked_agents.csv
    agents/maps_stats.csv

  Deferred files (Phase 5 only - very large, do not load early):
    matches/kills.csv
    matches/rounds_kills.csv

  Global reference (always loaded first before any joins):
    all_ids/all_teams_mapping.csv       Resolves team name changes across seasons
    all_ids/all_players_ids.csv         Resolves player IDs across seasons

  WARNING: Never join datasets on raw name strings.
           Always resolve to IDs first using the all_ids/ reference files.


================================================================================
  TECH STACK
================================================================================

  Data & ML   :  Python, Pandas, NumPy, Scikit-learn, XGBoost
  Backend     :  FastAPI
  Frontend    :  React
  Deployment  :  Vercel (frontend) + Render or Railway (backend)
  Versioning  :  Git & GitHub


================================================================================
  PROJECT STRUCTURE
================================================================================

  valorant-ml-project/
  |
  +-- data/                  Raw & cleaned datasets
  +-- notebooks/             EDA and model prototyping
  +-- src/
  |   +-- data/              Loading, cleaning, ID resolution
  |   +-- scoring/           Player score, chemistry, role balance
  |   +-- models/            ML training, evaluation, modifiers
  |   +-- recommendation/    Agent recommendation engine
  |   +-- simulator/         Custom match simulation logic
  +-- models/                Trained ML models (.joblib)
  +-- frontend/              React app
  +-- backend/               FastAPI app
  +-- app.py                 Main backend entry point
  +-- requirements.txt       Python dependencies
  +-- README.md              Public-facing GitHub documentation


================================================================================
  DEVELOPMENT PHASES
================================================================================

  Phase 1  ->  Data pipeline + individual player scoring + baseline prediction
  Phase 2  ->  Player comparison module (mechanical vs agent-adjusted)
  Phase 3  ->  Chemistry scoring + full roster-aware prediction (core innovation)
  Phase 4  ->  Agent recommendation module
  Phase 5  ->  Custom match simulator
  Phase 6  ->  React frontend + FastAPI backend + deployment

  Rule: Build Phase 1 + 2 as a polished MVP before touching Phase 3 onward.
        Correctness and completeness beat breadth every time.


================================================================================
  KNOWN GAPS & DEFERRED DECISIONS
================================================================================

  IGL identification
    Check if dataset includes IGL flags per player.
    If yes  ->  add leadership multiplier to chemistry score in Phase 3.
    If no   ->  defer and document as known gap.

  VCT 2026 live data
    Not available until end of season.
    Current approach: user inputs 2026 roster manually; system predicts
    using 2021-2025 historical player scores.
    Live data pipeline deferred to a future version.


================================================================================
  EXPECTED OUTCOMES
================================================================================

  - Roster-aware match prediction with full breakdown output
  - Individual player scoring across 5 years of VCT history
  - 1v1 player comparisons: mechanical skill and agent-adjusted modes
  - Agent composition recommendations tailored per player and opponent
  - Realistic custom 5v5 match simulations with any tier 1 players
  - Deployed web platform + clean GitHub repository

================================================================================