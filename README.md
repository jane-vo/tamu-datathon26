# Goofy Gamblers present: Self-improving poker bot utilizing Snowflake & Gemini

**TAMU Datathon Heads Up Challenge** • **Best Use of Snowflake** • **Best Use of Google AI Studio**

Heads-up Short Deck 5-Card Pot-Limit Omaha • Python

Most teams will make their agent by using one-shot programming. We built a loop that **measures** it. Every decision is logged to **Snowflake**, a fair-comparison query that shows where the bot leaks chips, and autonomously transferred to **Google AI Studio (Gemini),** which proposes fixes and writes tougher opponents. An improvement to the agent occurs only if it wins with a 95% confidence interval.

## 📖 Table of Contents

- [The bot](#the-bot)
- [The loop](#the-loop)
- [Snowflake: evidence layer](#snowflake-evidence-layer)
- [Google AI Studio (Gemini): improvement engine](#google-ai-studio-gemini-improvement-engine)
- [When a change is real](#when-a-change-is-real)
- [Results](#results)
- [Version history](#version-history)
- [Repository map](#repository-map)
- [Run it yourself](#run-it-yourself)
- [Honest limits](#honest-limits)
- [Security](#security)

## The bot

`agent.py` (v4) is a fast, crash-proof equity bot that models each opponent, drafted and tuned with the loop below. It beat v3 against all 12 test opponents, by +1.28 bb/deal on average:

- **numpy engine:** vectorised Omaha scoring (every 2-hole × 3-board combination is one table lookup), about 20× faster than pure Python. Each decision simulates hundreds of opponent hands instead of about 60.
- **Opponent range (particle filter):** at the start of a hand the bot samples possible opponent hands. Every opponent action then re-weights them by how likely *this* opponent is to take that action with that hand, using its measured bet, call, fold and re-raise rates. A maniac's bets barely narrow its range; a nit's bets narrow it a lot.
- **Expected-value decisions:** fold, call, half-pot and pot are compared by expected chips. Raises use the opponent's measured fold rate (tracked separately for facing a first bet and facing a re-raise), our equity against the hands that would call, and what happens when the strongest hands re-raise.
- **Fold rates by situation:** tracking those two fold rates separately was the biggest single gain over v3. It stopped the bot raising into strong hands.
- **Preflop:** keeps pots small. It never open-raises, and it calls using a formula fitted offline (`fit_preflop.py`) to Monte Carlo equity. Most of its edge comes after the flop.
- **Learns from hidden information:** an opponent's fold to our last bet, or a river check-behind, never shows up in a later observation, so the bot infers these from the hand result.
- **Built for the rules:**
  - The judge charges our thinking *plus* about 0.7 ms of request handling per call. The compute budget therefore adapts to how many requests the half needs (6.2 s minus that overhead, so 1.5–3.0 s of compute), and a safety valve switches to zero-compute play at 6.9 s of estimated use, before the 8 s bank could run out.
  - Any exception falls back to a legal check or fold, so a bug can never trigger the five-error forfeit. Without numpy, the v3 pure-Python bot plays instead.
  - No network and no writes outside `/tmp` at match time. Both outside tools run offline.

## The loop

```
  harness_log.py ──► logs/*.csv ──► snowflake_load.py ──► Snowflake (POKER_BOT)
   self-play, eps=0 / 0.1                                        │ analysis.sql
        ▲                                                        ▼
        │                                              gemini_helper.py analyze
   test with --param, --eps 0                                    │
        ▲                                                        ▼
        └──── keep only if CI > 0 ◄──── Gemini proposes setting changes
                    gemini_helper.py sparring ──► harder opponents (sparring/)
```

One command runs the whole evidence loop:

```bash
python gemini_helper.py optimize            # or: python autotune.py
```

1. **Data:** self-play against the baselines and the Gemini sparring bots, with 10% random actions (`harness_log.py`).
2. **Snowflake:** logs are uploaded and the fair-comparison queries from `analysis.sql` run on that iteration's data only.
3. **Gemini:** receives the results plus everything already tested, and must reply with strict JSON (up to 3 single-setting changes).
4. **Guardrails:** a whitelist and numeric bounds. The time budget, model coefficients and any code are off limits; refused proposals are logged.
5. **Testing:** each change is tested alone against the current bot on identical cards. A cheap *screen* must show a gain, then a *confirmation* on fresh seeds must have its 95% interval entirely above 0 (and no opponent significantly worse).
6. **Output:** adopted changes go to `agent_candidate.py`; `agent.py` is never edited unless `--apply` is passed, and then only if the tests pass.

Every proposal, adopted or not, is written to `logs/autotune_log.csv` and `logs/autotune_report.md`. Results from our runs are in the Results section.

**What this loop already caught:** in a test run, all three of Gemini's suggestions passed the cheap screen with small positive gains, and all three turned negative on fresh seeds. A single-stage test would have adopted them.

### Snowflake: evidence layer

- Every self-play decision becomes a row: street, position, hole cards, board, estimated equity, pot, price to call, action, and the hand's final result. Matches get their own table with confidence intervals and compute time.
- We play two kinds of runs: **eps = 0** (clean strength measurement) and **eps = 0.1**, where 10% of actions are replaced by a random legal action and flagged `EXPLORED = 1`.
- **Query 2 in `analysis.sql` is the key one.** Because the random actions are independent of the bot's preferences, comparing actions inside an equity bucket is an unbiased estimate of what each choice is worth. Ordinary logs can't tell you that.
- Snowflake's built-in assistant (CoCo) was blocked on our account, so we exported query results and drove the analysis from the terminal through Gemini instead.
- Loaded files move to `logs/loaded/` so nothing is uploaded twice.

### Google AI Studio (Gemini): improvement engine

1. **Sparring partners:** `gemini_helper.py sparring` has Gemini write opponents in different styles (bluffer, trapper, nit and so on) as ready-to-run bot files. They're wrapped so illegal or crashing output falls back to a safe action. The baselines are weak, so this is how we avoid overfitting to them.
2. **Tuning suggestions:** `gemini_helper.py analyze` sends the Snowflake results plus our tunable settings to Gemini and asks for the likely leaks, one concrete setting change for each, and how many hands are needed to confirm it.
3. **No change is taken on faith:** `autotune.py` runs the loop with guardrails. Gemini may change only eight whitelisted decision settings within set bounds (never the time settings), and a change must pass a screen and then a confirmation on fresh cards before it is kept. Every proposal and result is logged.

## When a change is real

- Matches are **duplicate** (both bots get the same cards), the same format as the tournament.
- A hand outcome has a standard deviation of about **25 bb per deal**, so small edges need **thousands of hands**. We keep a change only if its **95% confidence interval is entirely above 0**, it gains at least 0.15 bb/deal, and it **doesn't clearly lose to any single opponent**.
- One example of the process working: the first full judge run used 6.3 s of the 8 s bank. Measuring compute budgets showed that a small fixed budget cost strength (1.8 s lost 0.74 bb/deal against v3), while a bigger sample of opponent hands helped up to about 600. That led to v4's adaptive time budget.

## Results

**v3 → v4**, duplicate matches, bb/deal, 4 seeds × 1,000 deals per opponent (95% CI of the paired difference). v4 improved against all 12 opponents:

| Opponent | v3 (bb/deal) | v4 (bb/deal) | v4 − v3 [95% CI] |
|---|---|---|---|
| RandomBot | +7.35 | +9.48 | +2.13 [+1.28, +2.98] |
| CallingStation | +7.84 | +9.63 | +1.79 [+1.46, +2.13] |
| AlwaysPot | +6.12 | +7.53 | +1.41 [+0.65, +2.17] |
| AlwaysHalf | +11.00 | +12.29 | +1.29 [+0.66, +1.92] |
| Gemini bluffer | +9.42 | +10.28 | +0.87 [+0.13, +1.61] |
| Gemini maniac | +5.73 | +6.38 | +0.65 [−0.09, +1.40] |
| Gemini nit | +3.10 | +3.79 | +0.69 [+0.38, +1.01] |
| Gemini station | +8.86 | +10.66 | +1.81 [+1.42, +2.19] |
| Gemini trapper | +12.21 | +14.19 | +1.98 [+1.49, +2.47] |
| v3 itself | +0.06 | +1.05 | +0.99 [+0.53, +1.44] |
| v3, loose-aggressive variant | +1.29 | +1.73 | +0.44 [−0.10, +0.98] |
| v3, tight-aggressive variant | −0.64 | +0.64 | +1.28 [+0.87, +1.69] |
| **Mean** | **+6.03** | **+7.31** | **+1.28** |

In five real HTTP judge matches against v3 (1,000 deals each, `judge_engine.py`), v4 came out ahead every time: +0.54 to +0.71 bb/deal, pooled **+0.63 [+0.31, +0.95]**. Each single match is a CHOP, because one 1,000-deal match against a close copy of itself can't separate an edge that small. v4 also used less of the time bank: 5.4–6.3 s against v3's 6.1–7.3 s. Against the baselines v4 scored +9.65 bb/deal vs CallingStation and +6.45 vs AlwaysPot (single matches, each about ±1.6 bb/deal), using at most 6.7 s of the 8 s bank.

**Earlier Gemini tuning rounds (v1 settings; none adopted):**

| Change tested | Result (bb/deal) | Outcome |
|---|---|---|
| *`OPEN_PRE=(0.48, 0.60)`* | *-0.08 [-0.49, +0.34]* | *unconfirmed* |
| *`OPEN_POST=(0.65, 0.75)`* | *+0.02 [-0.39, +0.44]* | *unconfirmed* |
| *`BASE_MARGIN=0.02`* | *+0.35 [-0.08, +0.78], then -0.10 on replication* | *not reproduced* |
| *All three round-1 changes combined* | *+0.20 [-0.23, +0.62]* | *unconfirmed* |
| *`OPEN_PRE=(0.56, 0.62)`* | *-0.43 [-0.84, -0.02]* | *leans worse* |
| *`OPEN_POST=(0.65, 0.78)`* | *-0.45 [-0.86, -0.03]* | *leans worse* |
| *`RERAISE_POST=(0.72, 0.85)`* | *-0.70 [-1.27, -0.12]* | *leans worse* |
| *`RERAISE_PRE=(0.50, 0.64)`* | *-1.05 [-1.67, -0.43], screen only (9 opponents × 500 deals)* | *leans worse (not adopted)* |

Autotune run 1 (9 opponents, including 5 Gemini sparring bots): Gemini proposed RERAISE_PRE=(0.50, 0.64), citing a high average result for preflop raises at equity 0.50. That number came from single random deviations, not from a threshold policy, and the paired screen test lost 1.05 bb/deal, so it was rejected. In run 2 Snowflake was unreachable and Gemini returned no valid proposals. No Gemini tuning change was ever adopted: v4's gains came from the engine and opponent-model work, measured with the same paired tests.

Also measured and rejected while building v4: raising more preflop (lost to aggressive bots by giving up the postflop edge), raising premium hands preflop (−0.08 to −0.20 bb/deal), and a smaller fixed 1.8 s compute budget (−0.74 bb/deal against v3).

## Version history

- **v1, the basic agent:** the equity bot on its own, with no Snowflake or Gemini loop yet: a fitted preflop formula, Monte Carlo win chance against a random hand after the flop, pot-odds decisions and a simple opponent model. Duplicate matches, 1,000 deals per half: +7.37 bb/deal vs RandomBot, +7.13 vs CallingStation, +7.63 vs AlwaysPot, +12.58 vs AlwaysHalf. Judge run vs AlwaysPot: +7.03 [+5.45, +8.62], 0 errors.
- **v2, the loop:** added the Snowflake + Gemini pipeline around the same bot: decision logging with random exploration, the Snowflake upload and analysis queries, Gemini tuning suggestions and Gemini-written sparring bots. The loop's own code had not yet been tested or edited, and the bot's strategy stayed the same as v1.
- **v3:** our previous best version, a pure-Python bot and the baseline v4 is measured against in the Results table above. It is also v4's fallback if numpy is missing.
- **v4, the current bot:** numpy hand engine, opponent-range model, expected-value decisions and the adaptive time bank, described in The bot section.

## Repository map

| File | What it does |
|---|---|
| `agent.py` | The submitted bot, v4 (strategy between the marked sections; server code untouched) |
| `fit_preflop.py` | Offline fit of the preflop strength formula |
| `harness.py` | Server-free duplicate-match runner using the real `game.py` |
| `harness_log.py` | Plays matches and logs every decision to CSV (`--eps`, `--param`, `--version`) |
| `snowflake_load.py` | Creates the `POKER_BOT` tables and uploads the CSVs |
| `analysis.sql` | Starter queries, including the unbiased action comparison |
| `gemini_helper.py` | `analyze` (tuning suggestions) and `sparring` (opponent bots) via Gemini |
| `requirements-analysis.txt` | Offline-only tools. Not used by the bot image |
| `compare_versions.py` | Paired A/B test of two agent files on the same cards (also used by autotune) |
| `autotune.py` | Guarded tuning loop: Gemini proposes, paired tests decide, every result is logged |
| `envfile.py` | Loads .env without an extra package; a variable set in the shell wins |
| `.env.example` | Template for GEMINI_API_KEY and SNOWFLAKE_\* variables. Copy it to .env, which git ignores |
| `test_agent.py` | Template tests plus 4 of ours: scorer matches the official evaluator, run-outs never reuse cards, both fallbacks play legally |

## Run it yourself

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-analysis.txt      # offline analysis tools only
python -m pytest -q                                      # bot passes the template tests
cp .env.example .env                                     # then add your GEMINI_API_KEY
# judge matches: use http://127.0.0.1:5008, not localhost

# generate data (clean + exploratory), then load it
python harness_log.py --opponents random,calling_station,always_pot,always_half --seeds 3 --deals 500 --eps 0
python harness_log.py --opponents random,calling_station,always_pot,always_half --seeds 3 --deals 500 --eps 0.1
python snowflake_load.py        # reads SNOWFLAKE_* from .env or the shell
# then run analysis.sql in a worksheet (USE DATABASE POKER_BOT;)

# Gemini (needs GEMINI_API_KEY in .env)
python gemini_helper.py sparring --name bluffer --style "raises weak hands a lot but folds to big re-raises"
python gemini_helper.py analyze results.txt

# test a suggested change by hand, or run the whole guarded loop
python harness_log.py --opponents always_pot --deals 1000 --eps 0 --param CALL_MARGIN=0.03 --version margin03
python gemini_helper.py optimize          # full guarded loop (autotune.py)
```

## Honest limits

- The baseline bots are weak, so an easy win over them says little about stronger opponents. That's why we use Gemini-written sparring bots.
- The opponent-range model is only as good as its read. Early in a half, before there is data, it relies on assumed bet and fold rates.
- All timings come from one Windows laptop. Tournament hardware is unknown, so the time governor keeps a margin and has a zero-compute fallback.
- Hand results are noisy. Any claim here comes with its sample size, and we say so when an effect isn't confirmed.

## Security

API keys and Snowflake credentials live in a git-ignored `.env` file (template: `.env.example`) or in environment variables, never in the code. Logs and generated sparring bots are git-ignored too. Neither Snowflake nor Gemini is called by the bot during matches.
