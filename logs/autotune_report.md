# Autotune report: changes proposed by Gemini and what testing showed

_Last updated 2026-11-07 19:58._

**Method.** Each change is tested alone against the current bot on identical cards (paired, duplicate format). A cheap *screen* must show at least 0.15 bb/deal; the change is then re-tested on *fresh seeds* and adopted only if the 95% interval is entirely above 0. **Outcome** is *leans better* / *leans worse* when the pooled difference is at least 0.15 bb/deal in that direction, otherwise *minimal change*. Hand results are noisy (about 25 bb per deal), so small effects need many hands.

| # | Change tested | Result (bb/deal, 95% CI) | Tested on | Outcome | Decision |
|---|---|---|---|---|---|
| 1 | `RERAISE_PRE` (0.58, 0.64) → (0.5, 0.64) | -1.05 [-1.67, -0.43] | screen: 9 matches × 500 deals | leans worse | not adopted (failed screen) |

Rejected proposals are kept on purpose: they show the guardrails and the testing worked. Evidence source for the latest run: local CSV analysis.
