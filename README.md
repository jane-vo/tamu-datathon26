# Goofy Gamblers present: Self-improving poker bot utilizing Snowflake & Gemini

**TAMU Datathon Heads Up Challenge • Best Use of Snowflake • Best Use of Google AI Studio**

Heads-up Short Deck 5-Card Pot-Limit Omaha • Python


Most teams will make their agent by using one-shot programming. We built a loop that measures it. Every decision is logged to **Snowflake**, a fair-comparison query shows where the bot leaks chips, **Google AI Studio (Gemini)** proposes fixes and writes tougher opponents, and a change to the agent occurs only if it wins with a 95% confidence interval above zero.

## 📖 Table of Contents
- [The bot](#the-bot)
- [The loop](#the-loop-snowflake-gemini)
- [Snowflake: evidence layer](#snowflake-evidence-layer)
- [Google AI Studio (Gemini): improvement engine](#google-ai-studio-gemini-improvement-engine)
- [When a change is real](#when-a-change-is-real)

## The bot

`agent.py` (v4) is a fast, crash-proof equity bot that models each opponent (drafted with Claude, then tuned with the loop below).

- **numpy engine:** vectorised Omaha scoring (every 2-hole × 3-board combination is one table lookup), about 20× faster than pure Python. Each decision simulates hundreds of opponent hands instead of about 60.
- **Opponent range (particle filter):** at the start of a hand the bot samples possible opponent hands. Every opponent action then re-weights them by how likely *this* opponent is to take that action with that hand, using its measured bet, call, fold and re-raise rates. A maniac's bets barely narrow its range; a nit's bets narrow it a lot.
- **Expected-value decisions:** fold, call, half-pot and pot are compared by expected chips. Raises use the opponent's measured fold rate (tracked separately for facing a first bet and facing a re-raise), our equity against the hands that would call, and what happens when the strongest hands re-raise.
- **Preflop:** keeps pots small. It never open-raises, and it calls using a formula fitted offline (`fit_preflop.py`) to Monte Carlo equity. Most of its edge comes after the flop.
- **Built for the rules:**
  - The judge charges our thinking *plus* about 0.7 ms of request handling per call. The compute budget therefore adapts to how many requests the half needs, and a safety valve switches to zero-compute play before the 8 s bank could run out.
  - Any exception falls back to a legal check or fold, so a bug can never trigger the five-error forfeit. Without numpy, the v3 pure-Python bot plays instead.
  - No network and no writes outside `/tmp` at match time. Both outside tools run offline.

**v3 → v4**, duplicate matches, bb/deal, 4 seeds × 1000 deals per opponent (95% CI of the paired difference):

| Opponent | v3 | v4 | v4 − v3 |
|---|---|---|---|
| random | +7.35 | +9.48 | +2.13 [+1.28, +2.98] |
| calling_station | +7.84 | +9.63 | +1.79 [+1.46, +2.13] |
| always_pot | +6.12 | +7.53 | +1.41 [+0.65, +2.17] |
| always_half | +11.00 | +12.29 | +1.29 [+0.66, +1.92] |
| Gemini bluffer / maniac / nit | +9.42 / +5.73 / +3.10 | +10.28 / +6.38 / +3.79 | +0.87 / +0.65 / +0.69 |
| Gemini station / trapper | +8.86 / +12.21 | +10.66 / +14.19 | +1.81 / +1.98 |
| **v3 itself** | +0.06 | **+1.05** | **+0.99 [+0.53, +1.44]** |
| **Mean** | +6.03 | **+7.31** | **+1.28** |

In five real HTTP judge matches against v3 (1000 deals each), v4 came out ahead every time: +0.54 to +0.71 bb/deal, pooled +0.63 [+0.31, +0.95]. Each single match is a CHOP, because one 1000-deal match against a close copy of itself can't separate an edge that small. v4 also used less of the time bank: 5.4–6.3 s against v3's 6.1–7.3 s.

## The loop (Snowflake + Gemini)

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

Every proposal, adopted or not, is written to `logs/autotune_log.csv` and `logs/autotune_report.md`:

<!-- paste the table printed at the end of a run (or the table from logs/autotune_report.md) here -->

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
3. **Nothing is trusted on faith.** Each suggestion is run as `--param NAME=VALUE --version <label> --eps 0` and kept only if it clears the bar below.

## When a change is real

- Matches are **duplicate** (both bots get the same cards), the same format as the tournament.
- A hand outcome has a standard deviation of about **25 bb per deal**, so small edges need **thousands of hands**. We keep a change only if its 95% confidence interval is entirely above 0.
- One example of the process working: the first full judge run used 6.3 s of the 8 s bank. We tested a smaller compute budget and cut it to 2.5 s, which cost nothing measurable (+6.97 vs +6.91 bb/deal against AlwaysPot at 2.5 s vs 3.5 s). Dropping to 2.0 s did hurt, so we stopped there.
