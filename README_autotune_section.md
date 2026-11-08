## Autonomous tuning loop (`gemini_helper.py optimize`)

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
