# 🎯 VALORANT AI Analytics Platform
### Match Prediction, Roster Intelligence & Simulation System

> **🟢 CURRENT PROJECT STATUS**
> * **[✅ COMPLETED] Module 1 (Match Prediction Engine):** The data pipeline, historical XGBoost prediction engine, FastAPI backend, and dynamic Frontend UI are fully implemented and operational.
> * **[🛠️ IN PROGRESS] Module 2 (Roster Transfer Impact):** Currently architecting the simulation engine to evaluate the performance swing of theoretical roster changes.

---

## 🧠 Overview

The VALORANT AI Analytics Platform is a machine learning system designed to predict match outcomes using **dynamic roster-aware modeling**.

Unlike traditional esports models that treat teams as fixed units, this system evaluates individual player performance across multiple seasons and dynamically recalculates team strength based on current roster composition.

It incorporates:
- 🔄 Player transfers and roster changes
- 🤝 Team chemistry evolution over time
- 🗺️ Map-specific and side-specific strengths

to produce realistic, context-aware match predictions.

---

## ❌ Core Problem & ✅ Solution

### The Problem with Traditional Systems

Most esports prediction models assume: **"Team = fixed entity"**

But in reality, rosters change constantly.

### 📌 Example Scenario

| Year |          Team A |------------------|Team B 
|------|------------------------------------|--------------------------------------|
| 2024 | a, b, c, d, e |                    |v, w, x, y, z |
| 2025 | a, b, c, d, **g** *(g replaces e)* |**t, q**, x, y, z *(t, q replace v, w)* |

Traditional models still treat Team A as the same team — ignoring new players and chemistry shifts entirely.

### 💡 This System's Solution

- 📊 Player performance is tracked **individually** across all teams and seasons
- ⚗️ New players introduce a **chemistry penalty** on joining
- 📈 Chemistry improves **logarithmically** as players play more matches together
- 🔄 Final prediction uses **updated team strength**, not outdated historical data
- 🌱 Rookies get a **minimum performance floor** (tier 1 pros earn their spot)

---

## ⚙️ System Architecture

All modules are built on a shared core engine:

```
Player Data  →  Player Scores  →  Team Strength  →  Context  →  Prediction
```

**Context factors:**
- 🗺️ Map performance and side tendencies (attack vs defense)
- 🎭 Role balance across the roster
- 🤝 Team chemistry score
- 🎮 Match format (Bo1 / Bo3 / Bo5)
- 🏆 Tournament stage (group stage vs playoffs)

---

## 🚀 Modules

### 🟢 Module 1 — Match Prediction Engine

**Goal:** Predict match outcomes using current rosters.

**📥 Input**
- Team A (5 players) and Team B (5 players)
- Map pool
- Match format: Bo1 / Bo3 / Bo5
- Time window: last 6 months / 1 year / 2 years / all time

**⚙️ How It Works**
1. Calculate individual player scores for all 10 players
2. Aggregate into team strength scores
3. Apply chemistry adjustments, map advantages, and role balance
4. Apply match format modifier and tournament stage modifier
5. Feed all features into trained XGBoost model

**📤 Output (Full Breakdown)**
- Win probability % per team
- Player matchup analysis (who wins their individual lane)
- Map-by-map odds across the pool
- Recommended map bans per team
- Key factors driving the prediction *(e.g. "Team A has +15% chemistry advantage on Ascent")*

---

### 🟡 Module 2 — Roster Change Impact Analysis

**Goal:** Evaluate whether a roster change improves or weakens a team.

```
Original : Team(a, b, c, d, e)
Updated  : Team(a, b, c, d, x)
delta    = strength(a,b,c,d,x) - strength(a,b,c,d,e)
```

**📤 Output**
- Performance change (%)
- Chemistry impact of the swap
- Role balance shift
- Projected team strength trajectory as chemistry builds

---

### 🔥 Module 3 — Player Intelligence System

#### 📊 3A — Player Comparison

Compare any two players across mechanical skill, utility impact, map performance, and clutch ability.

| Mode | Metrics | Answers |
|------|---------|---------|
| 🎯 **Without Abilities** | ACS, K/D, HS%, accuracy | "Who is the better pure aimer?" |
| 🧠 **With Abilities** | All of the above + utility damage, assists, agent win rate | "Who is the better complete VALORANT player?" |

Filters supported: time window, map, opponent

#### ⚡ 3B — Clutch Probability System

Estimate win probability in 1v1 through 1v5 scenarios, factoring in mechanical skill, agent utility, map, and opponent context.

---

### 🟣 Module 4 — Agent Recommendation System

**Goal:** Suggest the optimal 5-agent composition for a team on a given map.

**📥 Input**
- Your team's 5 players
- Opponent team *(system fetches their most recent agent composition)*
- Map being played

**⚙️ Logic**
1. Rank each player's agents by historical performance on this map
2. Enforce role coverage: Duelist, Initiator, Controller, Sentinel, Flex
3. Counter-pick opponent's last composition where possible
4. Never sacrifice player comfort for a counter-pick

**📤 Output**
- Recommended agent per player
- Role each player fills
- Reasoning per recommendation
- Opponent composition it is designed to counter

---

### 🔵 Module 5 — Custom Match Simulator

**Goal:** Simulate a match between any two custom 5-player teams.

**📌 Example**
```
Team A: TenZ, Aspas, Derke, Chronicle, FNS
Team B: Yay, Something, Leo, Crashies, Boaster
```

**🤝 Chemistry Handling for Unknown Combinations**

Players who have never played together are fully allowed. Unknown combinations start at a **50% chemistry floor**.

> 💡 Rationale: all players in this dataset are tier 1 professionals. Even with zero shared history, individual skill still dominates — you can't rule out a team of the world's best.

**📤 Output**
- Win probability per team
- Per-player performance projections
- 📉 Weakest matchup *(your team's liability)*
- 📈 Strongest matchup *(your team's biggest advantage)*
- Recommended agent composition for both teams

---

## 🧮 Scoring System

### 🧍 Individual Player Score

Built from the player's full match history across all teams and seasons. **Score travels with the player, not the team.** Respects the user-selected time window.

| Sub-score | Description |
|-----------|-------------|
| `mechanical_score` | ACS, K/D, HS%, first blood rate — no agent influence |
| `utility_score` | Performance delta on utility-heavy agents |
| `map_scores` | Per-map win rate and stat performance |
| `side_scores` | Attack and defense win rate, normalized per map |
| `clutch_score` | Round win rate in 1vX situations |
| `playoff_score` | Performance specifically in elimination matches |
| `rookie_floor` | Applied for players with fewer than 10 tier 1 matches. Floor = bottom 25th percentile of T1 players, adjusts upward dynamically |

### 🤝 Team Chemistry Score

Measures how well a specific group of 5 players performs together.

**📉 Decay curve (logarithmic):**

| Matches Together | Chemistry Level |
|-----------------|-----------------|
| 0 – 5 matches | 60% of full chemistry |
| 5 – 15 matches | 80% of full chemistry |
| 15+ matches | 95 – 100% of full chemistry |

> **Minimum floor: 50%** for any combination — tier 1 pros don't fall apart on their first match together.

### 🎭 Role Balance Score

| Condition | Effect |
|-----------|--------|
| Missing a required role | ❌ Penalty applied |
| Full role coverage | ✅ Bonus applied |
| Player's best agent matches assigned role | ⭐ Additional bonus |

Required roles: **Duelist / Initiator / Controller / Sentinel / Flex**

### 🗺️ Map Performance Score

Per team, per map:
- Historical win rate
- ⚔️ Attack win rate *(normalized against global map side bias)*
- 🛡️ Defense win rate *(normalized against global map side bias)*

Used for: map odds, recommended bans, Bo3/Bo5 series prediction.

### 🎮 Format & Stage Modifiers

| Match Format | Effect |
|-------------|--------|
| Bo1 | Amplify map-specific scores, reduce chemistry weight |
| Bo3 | Balanced weights across all scores |
| Bo5 | Amplify chemistry, side balance, and consistency |

| Tournament Stage | Effect |
|-----------------|--------|
| Group stage | Standard weights |
| 🏆 Playoffs | `playoff_score` weighted at 1.3x |

---

## 🤖 Full Prediction Pipeline

```
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
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — VCT Champion Tour 2021–2025 |
| Local path | `D:/archive/` |
| Size | ~1.3 GB across 5 year folders |

### 🔑 Primary Files *(load first, used in Phase 1–3)*

| File | Purpose |
|------|---------|
| `matches/scores.csv` | Match results — **target variable** |
| `matches/overview.csv` | Player stats per match — **key features** |
| `matches/maps_played.csv` | Maps played per match |
| `matches/maps_scores.csv` | Map-level scores and side data |
| `matches/team_mapping.csv` | Player-to-team links — **key for chemistry** |
| `players_stats/players_stats.csv` | Aggregated player stats per season |

### 📁 Secondary Files *(Phase 4+)*
- `agents/agents_pick_rates.csv`
- `agents/teams_picked_agents.csv`
- `agents/maps_stats.csv`

### ⏳ Deferred Files *(Phase 5 only — very large, do not load early)*
- `matches/kills.csv`
- `matches/rounds_kills.csv`

### 🗂️ Global Reference *(always load before any joins)*
- `all_ids/all_teams_mapping.csv` — resolves team name changes across seasons
- `all_ids/all_players_ids.csv` — resolves player IDs across seasons

> ⚠️ **WARNING:** Never join datasets on raw name strings. Always resolve to IDs first using the `all_ids/` reference files.

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🐍 Data & ML | Python, Pandas, NumPy, Scikit-learn, XGBoost |
| ⚡ Backend | FastAPI |
| ⚛️ Frontend | React |
| ☁️ Deployment | Vercel (frontend) + Render / Railway (backend) |
| 🔧 Versioning | Git & GitHub |

---

## 🧱 Project Structure

```
valorant-ml-project/
│
├── data/                  Raw & cleaned datasets
├── notebooks/             EDA and model prototyping
├── src/
│   ├── data/              Loading, cleaning, ID resolution
│   ├── scoring/           Player score, chemistry, role balance
│   ├── models/            ML training, evaluation, modifiers
│   ├── recommendation/    Agent recommendation engine
│   └── simulator/         Custom match simulation logic
├── models/                Trained ML models (.joblib)
├── frontend/              React app
├── backend/               FastAPI app
├── app.py                 Main backend entry point
├── requirements.txt       Python dependencies
└── README.md              This file
```

---

## 📈 Development Phases

| Phase | Focus |
|-------|-------|
| ✅ Phase 1 | Data pipeline + individual player scoring + baseline prediction |
| ✅ Phase 2 | Player comparison module (mechanical vs agent-adjusted) |
| 🔥 Phase 3 | Chemistry scoring + full roster-aware prediction *(core innovation)* |
| 🟣 Phase 4 | Agent recommendation module |
| 🔵 Phase 5 | Custom match simulator |
| 🚀 Phase 6 | React frontend + FastAPI backend + deployment |

> **Rule:** Build Phase 1 + 2 as a polished MVP before touching Phase 3 onward. Correctness and completeness beat breadth every time.

---

## ⚠️ Known Gaps & Deferred Decisions

**🎙️ IGL Identification**
Check if the dataset includes IGL flags per player. If yes, add a leadership multiplier to chemistry score in Phase 3. If no, defer and document as a known gap.

**📡 VCT 2026 Live Data**
Not available until end of season. Current approach: user inputs 2026 roster manually; system predicts using 2021–2025 historical player scores. Live data pipeline deferred to a future version.

---

## 🏆 Expected Outcomes

- 🎯 Roster-aware match prediction with full breakdown output
- 📊 Individual player scoring across 5 years of VCT history
- ⚔️ 1v1 player comparisons: mechanical skill and agent-adjusted modes
- 🧩 Agent composition recommendations tailored per player and opponent
- 🎮 Realistic custom 5v5 match simulations with any tier 1 players
- 🌐 Deployed web platform + clean GitHub repository
