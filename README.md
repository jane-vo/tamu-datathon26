# Goofy Gamblers present: Self-improving poker bot utilizing Snowflake & Gemini

**TAMU Datathon Heads Up Challenge • Best Use of Snowflake • Best Use of Google AI Studio**

Heads-up Short Deck 5-Card Pot-Limit Omaha • Python


Most teams will make their agent by using one-shot programming. We built a loop that measures it. Every decision is logged to **Snowflake**, a fair-comparison query shows where the bot leaks chips, **Google AI Studio (Gemini)** proposes fixes and writes tougher opponents, and a change to the agent occurs only if it wins with a 95% confidence interval above zero.

---

## The bot

`agent.py` is a small, fast, crash-proof equity bot (drafted with Claude, then tuned with the loop below).

- **Preflop:** a formula fitted offline (`fit_preflop.py`) to Monte Carlo equity over thousands of random hands. It costs microseconds, so the time bank is saved for postflop.
- **Flop, turn, river:** Monte Carlo win chance vs a random hand under the exact 2-hole + 3-board Omaha rule, compared to pot odds to fold, call, bet half-pot or bet pot.
- **Opponent model:** tracks raise frequency and fold-to-raise within a half, then calls wider against maniacs and bluffs more against folders.
- **Built for the rules:**
  - A time governor spreads about 2.5 s of our own compute across each half and spends more on big pots (`BANK_BUDGET_S`).
  - Any exception falls back to a legal check or fold, so a bug can never trigger the five-error forfeit.
  - No network and no writes outside `/tmp` at match time. Both outside tools run offline.

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

### Snowflake: the evidence layer
- Every self-play decision becomes a row: street, position, hole cards, board, estimated equity, pot, price to call, action, and the hand's final result. Matches get their own table with confidence intervals and compute time.
- We play two kinds of runs: **eps = 0** (clean strength measurement) and **eps = 0.1**, where 10% of actions are replaced by a random legal action and flagged `EXPLORED = 1`.
- **Query 2 in `analysis.sql` is the key one.** Because the random actions are independent of the bot's preferences, comparing actions inside an equity bucket is an unbiased estimate of what each choice is worth. Ordinary logs can't tell you that.
- Snowflake's built-in assistant (CoCo) was blocked on our account, so we exported query results and drove the analysis from the terminal through Gemini instead.
- Loaded files move to `logs/loaded/` so nothing is uploaded twice.

### Google AI Studio (Gemini): the improvement engine
1. **Sparring partners:** `gemini_helper.py sparring` has Gemini write opponents in different styles (bluffer, trapper, nit and so on) as ready-to-run bot files. They're wrapped so illegal or crashing output falls back to a safe action. The baselines are weak, so this is how we avoid overfitting to them.
2. **Tuning suggestions:** `gemini_helper.py analyze` sends the Snowflake results plus our tunable settings to Gemini and asks for the likely leaks, one concrete setting change for each, and how many hands are needed to confirm it.
3. **Nothing is trusted on faith.** Each suggestion is run as `--param NAME=VALUE --version <label> --eps 0` and kept only if it clears the bar below.

## How we decide a change is real

- Matches are **duplicate** (both bots get the same cards), the same format as the tournament.
- A hand outcome has a standard deviation of about **25 bb per deal**, so small edges need **thousands of hands**. We keep a change only if its 95% confidence interval is entirely above 0.
- One example of the process working: the first full judge run used 6.3 s of the 8 s bank. We tested a smaller compute budget and cut it to 2.5 s, which cost nothing measurable (+6.97 vs +6.91 bb/deal against AlwaysPot at 2.5 s vs 3.5 s). Dropping to 2.0 s did hurt, so we stopped there.
