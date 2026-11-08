"""
autotune.py - ONE command that runs a guard-railed, evidence-driven tuning loop.

    python gemini_helper.py optimize            # same thing (gemini_helper hands over to this file)
    python autotune.py --help

Each iteration:
  1. DATA      play self-play matches with 10% random actions (harness_log.py) using the current settings
  2. SNOWFLAKE upload the logs (snowflake_load.py), run the fair-comparison queries from analysis.sql on
               just this iteration's data. If Snowflake is unavailable it falls back to the same
               analysis on the local CSVs and says so in the report.
  3. GEMINI    receives the results + the list of everything already tested, and must answer with strict
               JSON: up to 3 single-setting changes
  4. GUARDRAILS every proposal is checked against a whitelist and numeric bounds. Anything else
               (time budget, model coefficients, unknown names, out-of-range values) is refused and logged.
  5. TEST      each change is tested ALONE against the current bot on identical cards (paired):
                 screen   -> a cheap run; must show a gain of at least --min-effect bb/deal
                 confirm  -> a fresh set of seeds the screen never saw; the 95% interval must be entirely
                             above 0 AND the gain at least --min-effect
  6. LOG       every proposal is written to logs/autotune_log.csv and logs/autotune_report.md
               (a README-ready table). Nothing is ever hidden: rejected ideas stay in the log.
  7. OUTPUT    adopted changes go into agent_candidate.py (a copy). Your agent.py is NOT touched unless you
               pass --apply, and even then the tests must pass or the backup is restored.

Needs (all already in your repo): agent.py, harness.py, harness_log.py, compare_versions.py,
snowflake_load.py, analysis.sql, baseline-bots/. Optional: sparring/*.py (opponents written by Gemini).

Environment: GEMINI_API_KEY (unless --proposals-file), SNOWFLAKE_* (unless --no-snowflake).
Put them in the .env file in the repo root (git-ignored) or set them in your shell.
"""
import argparse, ast, csv, datetime, difflib, glob, json, os, re, shutil, subprocess, sys, time, uuid

import envfile
envfile.load()                     # GEMINI_API_KEY / SNOWFLAKE_* from .env (shell variables win)

import compare_versions as C
import harness as H

# ---------------------------------------------------------------------------
# Guardrails: the ONLY settings the loop may change, and how far.
# pair = (half-pot threshold, pot threshold) on equity; first must be <= second.
# ---------------------------------------------------------------------------
TUNABLES = {
    "OPEN_PRE":     {"kind": "pair",   "lo": 0.45, "hi": 0.65, "cap": 0.80},
    "OPEN_POST":    {"kind": "pair",   "lo": 0.50, "hi": 0.80, "cap": 0.92},
    "RERAISE_PRE":  {"kind": "pair",   "lo": 0.50, "hi": 0.70, "cap": 0.85},
    "RERAISE_POST": {"kind": "pair",   "lo": 0.55, "hi": 0.85, "cap": 0.95},
    "BASE_MARGIN":  {"kind": "scalar", "lo": 0.00, "hi": 0.10},
    "BLUFF_RATE":   {"kind": "scalar", "lo": 0.00, "hi": 0.40},
}
# Deliberately NOT tunable here: BANK_BUDGET_S (a bad value = time-bank forfeit), MAX_DECISION_MS,
# MIN_SAMPLES, MAX_SAMPLES, PREFLOP_COEF, and any code.

BASELINES = ("random", "calling_station", "always_pot", "always_half")

GAME = """Game: heads-up Short Deck (36 cards, 6..A) 5-card Pot-Limit Omaha, 80-chip stacks (40 big blinds, big blind = 2 chips).
Hand = exactly 2 hole cards + exactly 3 board cards. Short-deck ranking: flush beats full house, three of a kind beats a straight.
Actions: FOLD, CHECK_CALL, RAISE_HALF (half pot), RAISE_POT. Matches are duplicate (same cards both seats)."""

NOTES = """How the settings are used (only propose a change to the setting that GOVERNS the situation you cite):
- OPEN_PRE / OPEN_POST = [half-pot threshold, pot threshold] on equity; used ONLY when there is nothing to call (to_call = 0).
  Preflop, the small blind always faces to_call = 1, so OPEN_PRE only matters for the big blind's option after a limp.
- RERAISE_PRE / RERAISE_POST = same shape; used only when FACING a bet and considering a raise.
- BASE_MARGIN = extra equity required on top of pot odds to CALL a bet (it grows with raises and bet size).
- BLUFF_RATE = chance to bet a weak hand when the opponent folds a lot.
- Evidence columns: AVG_NET_BB is the average result in big blinds of the WHOLE hand after taking that action.
  EXPLORED=1 rows had a random action, so comparing actions inside the same situation is unbiased.
- A change only affects hands where its setting applies, so its effect per deal is small. Be skeptical of small N."""

LOG_COLS = ["timestamp", "run_id", "iteration", "setting", "old_value", "new_value", "reason", "stage_reached",
            "screen_diff", "screen_lo", "screen_hi", "screen_matches", "confirm_diff", "confirm_lo", "confirm_hi",
            "confirm_matches", "deals_per_match", "by_opponent", "outcome", "decision", "min_effect",
            "evidence_source", "agent_file"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Reading / writing the settings block of agent.py (comments are preserved)
# ---------------------------------------------------------------------------
def _setting_re(name):
    return re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]+?)(\s*(?:#.*)?)$", re.M)


def read_settings(text):
    found = {}
    for name in TUNABLES:
        m = _setting_re(name).search(text)
        if not m:
            raise SystemExit(f"agent file has no line like '{name} = ...'. autotune expects the tunable-settings "
                             f"block that came with agent.py v1 (names: {', '.join(TUNABLES)}).")
        found[name] = ast.literal_eval(m.group(2).strip())
    return found


def apply_settings(text, changes):
    for name, value in changes.items():
        pat = _setting_re(name)
        if len(pat.findall(text)) != 1:
            raise ValueError(f"expected exactly one '{name} = ...' line")
        text = pat.sub(lambda m: m.group(1) + repr(value) + m.group(3), text, count=1)
    return text


def validate(name, value):
    """Return the cleaned value, or raise ValueError with the reason it was refused."""
    if name not in TUNABLES:
        raise ValueError(f"'{name}' is not a tunable setting")
    spec = TUNABLES[name]
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except Exception:
            raise ValueError(f"cannot read value {value!r}")
    if spec["kind"] == "scalar":
        if isinstance(value, (list, tuple)) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a single number")
        x = round(float(value), 3)
        if not spec["lo"] <= x <= spec["hi"]:
            raise ValueError(f"{x} outside allowed range [{spec['lo']}, {spec['hi']}]")
        return x
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ValueError("expected two numbers [half_pot_threshold, pot_threshold]")
    a, b = round(float(value[0]), 3), round(float(value[1]), 3)
    if not spec["lo"] <= a <= spec["hi"]:
        raise ValueError(f"first value {a} outside allowed range [{spec['lo']}, {spec['hi']}]")
    if not a <= b <= spec["cap"]:
        raise ValueError(f"second value {b} must be between the first ({a}) and {spec['cap']}")
    return (a, b)


# ---------------------------------------------------------------------------
# History (CSV) and the README-ready report
# ---------------------------------------------------------------------------
def load_history(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_history(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in LOG_COLS})


def fmt_res(d, lo, hi):
    return f"{float(d):+.2f} [{float(lo):+.2f}, {float(hi):+.2f}]"


def fmt_val(v):
    return repr(v) if not isinstance(v, str) else v


def write_report(rows, path, min_effect):
    lines = ["# Autotune report: changes proposed by Gemini and what testing showed", "",
             f"_Last updated {datetime.datetime.now():%Y-%m-%d %H:%M}._", "",
             "**Method.** Each change is tested alone against the current bot on identical cards (paired, duplicate "
             "format). A cheap *screen* must show at least "
             f"{min_effect:g} bb/deal; the change is then re-tested on *fresh seeds* and adopted only if the 95% "
             "interval is entirely above 0. **Outcome** is *leans better* / *leans worse* when the pooled difference "
             f"is at least {min_effect:g} bb/deal in that direction, otherwise *minimal change*. Hand results are "
             "noisy (about 25 bb per deal), so small effects need many hands.", "",
             "| # | Change tested | Result (bb/deal, 95% CI) | Tested on | Outcome | Decision |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        change = f"`{r['setting']}` {r['old_value']} → {r['new_value']}"
        if r.get("confirm_diff", "") != "":
            res, on = fmt_res(r["confirm_diff"], r["confirm_lo"], r["confirm_hi"]), \
                f"confirm: {r['confirm_matches']} matches × {r['deals_per_match']} deals, fresh seeds"
        elif r.get("screen_diff", "") != "":
            res, on = fmt_res(r["screen_diff"], r["screen_lo"], r["screen_hi"]), \
                f"screen: {r['screen_matches']} matches × {r['deals_per_match']} deals"
        else:
            res, on = "n/a", "not tested"
        lines.append(f"| {i} | {change} | {res} | {on} | {r.get('outcome', '')} | {r.get('decision', '')} |")
    table = "\n".join(lines[lines.index("| # | Change tested | Result (bb/deal, 95% CI) | Tested on | Outcome | Decision |"):])
    src = rows[-1].get("evidence_source", "") if rows else ""
    lines += ["", "Rejected proposals are kept on purpose: they show the guardrails and the testing worked."
              + (f" Evidence source for the latest run: {src}." if src else "")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return table


# ---------------------------------------------------------------------------
# Step 1-2: data generation and the evidence text (Snowflake, with local fallback)
# ---------------------------------------------------------------------------
def run(cmd, timeout):
    """Run a command WITHOUT a shell; fail loudly (the old version silently ignored failures)."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd[:4])} ...\n    " + "\n    ".join(tail))
    return p.stdout


def generate_data(a, agent_path, opp_specs, accepted, folder, version, seed):
    os.makedirs(folder, exist_ok=True)
    cmd = [sys.executable, "harness_log.py", "--agent", agent_path, "--opponents", ",".join(opp_specs),
           "--deals", str(a.gen_deals), "--eps", "0.1", "--version", version, "--seed", seed, "--out", folder]
    for k, v in accepted.items():
        cmd += ["--param", f"{k}={v!r}"]
    run(cmd, timeout=3600)
    files = glob.glob(os.path.join(folder, "decisions_*.csv"))
    if not files:
        raise RuntimeError("data generation produced no decisions file")


def split_queries(sql_text):
    """[(label, sql)] parsed line by line, so semicolons inside comments cannot break it."""
    out, label, buf = [], None, []
    for line in sql_text.splitlines():
        s = line.strip()
        if s.startswith("--"):
            m = re.match(r"--\s*(\d+[a-z]?)\)", s)
            if m:
                label = m.group(1)
            continue
        if not s:
            continue
        buf.append(line)
        if s.endswith(";"):
            out.append((label, "\n".join(buf).rstrip().rstrip(";")))
            buf = []
    return out


def table_text(title, cols, rows, limit=90):
    lines = [f"### {title}", " | ".join(cols)]
    for r in rows[:limit]:
        lines.append(" | ".join("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r))
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows not shown)")
    return "\n".join(lines)


def snowflake_evidence(folder, version, labels=("2a", "2b", "2d")):
    run([sys.executable, "snowflake_load.py", "--dir", folder], timeout=1800)
    import snowflake_load as SL
    conn = SL.connect()
    try:
        cur = conn.cursor()
        cur.execute(f"USE DATABASE {os.environ.get('SNOWFLAKE_DATABASE', 'POKER_BOT')}")
        cur.execute(f"USE SCHEMA {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}")
        cur.execute("CREATE OR REPLACE TEMPORARY VIEW DECISIONS_CUR AS SELECT * FROM DECISIONS "
                    f"WHERE BOT_VERSION = '{version}'")  # only THIS iteration's data
        blocks = []
        for label, sql in split_queries(open("analysis.sql", encoding="utf-8").read()):
            if label in labels:
                cur.execute(re.sub(r"\bDECISIONS\b", "DECISIONS_CUR", sql))
                blocks.append(table_text(f"Snowflake query {label}", [c[0] for c in cur.description], cur.fetchall()))
        if not blocks:
            raise RuntimeError("no queries found in analysis.sql")
        return "\n\n".join(blocks)
    finally:
        conn.close()


def local_evidence(folder, min_n=20):
    """Same analysis as queries 2a / 2b / 2d, computed from the local CSVs (used if Snowflake is unavailable)."""
    rows = []
    for p in glob.glob(os.path.join(folder, "**", "decisions_*.csv"), recursive=True):
        with open(p, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if r["EXPLORED"] == "1" and r["EQUITY"] != ""]

    def agg(sel, bucket, title):
        groups = {}
        for r in rows:
            if sel(r):
                key = (r["STREET"], bucket(float(r["EQUITY"])), r["ACTION"])
                groups.setdefault(key, []).append(int(r["HAND_NET_CHIPS"]) / 2.0)
        out = [(k[0], k[1], k[2], len(v), round(sum(v) / len(v), 2)) for k, v in sorted(groups.items()) if len(v) >= min_n]
        return table_text(title, ["STREET", "EQUITY_BUCKET", "ACTION", "N", "AVG_NET_BB"], out)

    b10 = lambda e: int(e * 10) / 10
    b20 = lambda e: int(e * 20) / 20
    return "\n\n".join([
        agg(lambda r: int(r["TO_CALL"]) > 0, b10, "Local query 2a (facing a bet)"),
        agg(lambda r: int(r["TO_CALL"]) == 0, b10, "Local query 2b (nothing to call)"),
        agg(lambda r: 0.45 <= float(r["EQUITY"]) <= 0.85, b20, "Local query 2d (near-threshold, both spots)"),
    ])


# ---------------------------------------------------------------------------
# Step 3: Gemini (strict JSON) and proposal parsing
# ---------------------------------------------------------------------------
def gemini_json(prompt):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in the .env file (or your shell), or pass --proposals-file.")
    from google import genai
    client = genai.Client(api_key=key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    last = None
    for attempt in range(3):
        try:
            try:
                from google.genai import types
                cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
                return client.models.generate_content(model=model, contents=prompt, config=cfg).text
            except (ImportError, TypeError, AttributeError):
                return client.models.generate_content(model=model, contents=prompt).text
        except Exception as exc:  # rate limit, network, bad model name ...
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after 3 tries: {last}. If the model name is wrong, set GEMINI_MODEL.")


def parse_proposals(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            raise ValueError("no JSON object in the reply")
        data = json.loads(text[i:j + 1])
    items = data.get("proposals", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("'proposals' must be a list")
    return items


def build_prompt(current, evidence, history, max_n):
    allowed = "\n".join(
        f"- {n}: " + (f"[half, pot] with {s['lo']} <= half <= {s['hi']} and half <= pot <= {s['cap']}" if s["kind"] == "pair"
                      else f"a single number in [{s['lo']}, {s['hi']}]") for n, s in TUNABLES.items())
    tested = "\n".join(
        f"- {r['setting']} {r['old_value']} -> {r['new_value']}: "
        + (fmt_res(r['confirm_diff'], r['confirm_lo'], r['confirm_hi']) if r.get("confirm_diff", "") != ""
           else fmt_res(r['screen_diff'], r['screen_lo'], r['screen_hi']) if r.get("screen_diff", "") != "" else "not tested")
        + f" bb/deal vs the current bot, outcome: {r.get('outcome', '')}"
        for r in history[-40:] if r.get("outcome") not in ("", "n/a")) or "- (nothing yet)"
    return f"""You are helping tune a poker bot. {GAME}

The bot estimates its win chance vs a random hand, compares it with pot odds and uses these settings (current values):
{json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in current.items()})}

{NOTES}

Evidence from our own self-play logs (this iteration's data only):
{evidence[:24000]}

Changes ALREADY TESTED, with the measured effect vs the current bot (do not propose these again; learn from them):
{tested}

Task: propose at most {max_n} changes, each to ONE setting, that the evidence supports. Answer with ONLY this JSON:
{{"proposals": [{{"setting": "<name>", "value": <number or [half, pot]>, "reason": "<one sentence citing the evidence row>"}}]}}
Allowed settings and ranges (anything else will be refused):
{allowed}
If the evidence does not support any change, answer {{"proposals": []}}. Evidence from weak hands says nothing about a
threshold at the boundary, and small N is noise; do not guess."""


# ---------------------------------------------------------------------------
# Step 5: paired testing
# ---------------------------------------------------------------------------
def paired_stage(base_mod, var_mod, opps, deals, seeds, prefix, cache, label):
    diffs, by_opp, n_matches, loss_opp = [], {}, 0, ""
    for opp_name, opp_mod in opps:
        per, opp_diffs = [], []
        for k in range(seeds):
            seed = f"{prefix}-{opp_name}-{k}"
            if (prefix, opp_name, k) not in cache:                  # the base result is shared by all proposals
                cache[(prefix, opp_name, k)] = C.deal_results(base_mod, opp_mod, deals, seed)
            var = C.deal_results(var_mod, opp_mod, deals, seed)
            d = [v - b for v, b in zip(var, cache[(prefix, opp_name, k)])]
            diffs += d
            opp_diffs += d
            per.append(sum(d) / len(d))
            n_matches += 1
        by_opp[opp_name] = sum(per) / len(per)
        o_mu, o_se = C.mean_se(opp_diffs)
        if o_mu + 1.96 * o_se < 0 and not loss_opp:      # whole 95% interval below 0 for this opponent
            loss_opp = opp_name
        log(f"    {label} vs {opp_name:16s} diff {by_opp[opp_name]:+.3f} bb/deal")
    mu, se = C.mean_se(diffs)
    return {"d": mu, "lo": mu - 1.96 * se, "hi": mu + 1.96 * se, "matches": n_matches,
            "by_opp": ";".join(f"{k}:{v:+.2f}" for k, v in by_opp.items()), "loss_opp": loss_opp}


def classify(d, min_effect):
    return "leans better" if d >= min_effect else ("leans worse" if d <= -min_effect else "minimal change")


# ---------------------------------------------------------------------------
# Candidate file, verification, optional apply
# ---------------------------------------------------------------------------
def build_candidate(agent_path, accepted):
    text = open(agent_path, encoding="utf-8").read()
    new = apply_settings(text, accepted)
    cand = os.path.splitext(agent_path)[0] + "_candidate.py"
    compile(new, cand, "exec")
    open(cand, "w", encoding="utf-8").write(new)
    diff = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), agent_path, cand, n=0))
    return cand, diff


def verify_candidate(cand):
    """Smoke test: plays 100 deals; the harness raises if the bot ever returns an illegal action."""
    mod = H.load_agent(cand, "at_candidate")
    opp_name, opp_mod = C.load_opp("always_pot")
    t0 = time.time()
    res = C.deal_results(mod, opp_mod, 100, "verify-0")
    return sum(res) / len(res), time.time() - t0


def apply_candidate(agent_path, cand):
    bak = f"{agent_path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(agent_path, bak)
    shutil.copy(cand, agent_path)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        shutil.copy(bak, agent_path)
        log("pytest FAILED, so agent.py was restored from the backup. Output tail:")
        print("\n".join((p.stdout + p.stderr).strip().splitlines()[-10:]))
        return False
    log(f"pytest passed. {agent_path} updated (backup: {bak}). Now run a judge match before submitting.")
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="agent.py", help="bot file to tune (default agent.py)")
    p.add_argument("--iters", type=int, default=2, help="how many propose/test rounds (default 2)")
    p.add_argument("--max-proposals", type=int, default=3, help="max changes per round (default 3)")
    p.add_argument("--opponents", default=",".join(BASELINES), help="baseline names or .py paths")
    p.add_argument("--no-sparring", action="store_true", help="do not add sparring/*.py to the opponents")
    p.add_argument("--make-sparring", type=int, default=0, help="ask Gemini to write N sparring bots first")
    p.add_argument("--deals", type=int, default=1000, help="deals per test match (default 1000)")
    p.add_argument("--screen-seeds", type=int, default=2, help="seeds per opponent in the screen (default 2)")
    p.add_argument("--confirm-seeds", type=int, default=3, help="seeds per opponent in the confirm (default 3)")
    p.add_argument("--gen-deals", type=int, default=400, help="deals per exploration match for the logs (default 400)")
    p.add_argument("--min-effect", type=float, default=0.15, help="smallest gain that counts, bb/deal (default 0.15)")
    p.add_argument("--no-snowflake", action="store_true", help="skip Snowflake; analyze the local CSVs instead")
    p.add_argument("--proposals-file", help="JSON file of proposals instead of calling Gemini (same guardrails)")
    p.add_argument("--retest", action="store_true", help="allow re-testing a change already in the log")
    p.add_argument("--apply", action="store_true", help="after verification, copy the candidate over the agent file (runs pytest, restores on failure)")
    p.add_argument("--quick", action="store_true", help="small preset for a fast demo run (about 10 minutes)")
    p.add_argument("--seed", default=None, help="base seed text (default: random per run, so every run uses fresh cards)")
    p.add_argument("--out", default="logs", help="log folder (default logs)")
    return p


def main(argv=None, llm=None):
    for stream in (sys.stdout, sys.stderr):        # Windows consoles/pipes default to cp1252 and choke on → ×
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    if a.quick:
        a.deals, a.screen_seeds, a.confirm_seeds, a.gen_deals, a.iters = 500, 1, 2, 200, 1
    if a.apply and os.path.basename(a.agent) != "agent.py":
        raise SystemExit("--apply only works with agent.py (the test suite checks that file).")
    llm = llm or gemini_json
    run_id = time.strftime("%m%d%H%M") + uuid.uuid4().hex[:3]
    seed0 = a.seed or run_id
    os.makedirs(a.out, exist_ok=True)
    csv_path, md_path = os.path.join(a.out, "autotune_log.csv"), os.path.join(a.out, "autotune_report.md")

    # ---- preflight -------------------------------------------------------
    if not os.path.exists(a.agent):
        raise SystemExit(f"{a.agent} not found. Run from the repo root.")
    base_text = open(a.agent, encoding="utf-8").read()
    current = read_settings(base_text)
    specs = [s.strip() for s in a.opponents.split(",") if s.strip()]
    if a.make_sparring:
        try:
            import gemini_helper as G
            styles = ["bluffer: raises weak hands a lot but folds to big re-raises",
                      "trapper: checks strong hands and calls, then raises the turn or river",
                      "nit: only continues with strong hands and rarely bluffs"]
            for i in range(min(a.make_sparring, len(styles))):
                name, style = styles[i].split(":")[0], styles[i]
                code = G.extract_code(G.ask(f"Write a poker opponent. {GAME}\n{G.OBS}\nStyle: {style}\n"
                                          "Return ONLY a python code block with `def decide(obs, memory):` returning an Action "
                                          "(Action.FOLD is only legal when obs.to_call > 0; use memory['rng'] for randomness). "
                                          "No file or network access; under 80 lines."))
                log(f"wrote {G.build_bot(code, name)} (read it before trusting it)")
        except Exception as exc:
            log(f"could not create sparring bots: {exc}")
    if not a.no_sparring:
        specs += sorted(glob.glob(os.path.join("sparring", "*.py")))
    for s in specs:
        path = s if s.endswith(".py") else os.path.join("baseline-bots", f"baseline_{s}.py")
        if not os.path.exists(path):
            raise SystemExit(f"opponent file missing: {path}")
    opps = [C.load_opp(s) for s in specs]
    log(f"run {run_id} | agent {a.agent} | opponents: {', '.join(n for n, _ in opps)}")
    log(f"current settings: {current}")

    accepted, new_rows = {}, []
    evidence_src = "local CSV analysis" if a.no_snowflake else "Snowflake"
    try:
        for it in range(1, a.iters + 1):
            log(f"===== iteration {it}/{a.iters} =====")
            history = load_history(csv_path)
            base_settings = {**current, **accepted}

            # ---- proposals ------------------------------------------------
            if a.proposals_file:
                raw = open(a.proposals_file, encoding="utf-8").read()
                evidence, src = "(proposals supplied from a file; no analysis)", "proposals file"
            else:
                version, folder = f"at{run_id}i{it}", os.path.join(a.out, "autotune", f"{run_id}_i{it}")
                log("step 1: generating self-play logs with 10% random actions ...")
                generate_data(a, a.agent, specs, accepted, folder, version, f"{seed0}-gen{it}")
                src = evidence_src
                evidence = None
                if not a.no_snowflake:
                    try:
                        log("step 2: uploading to Snowflake and running the fair-comparison queries ...")
                        evidence = snowflake_evidence(folder, version)
                    except Exception as exc:
                        log(f"Snowflake unavailable ({str(exc).splitlines()[0][:150]}); using local analysis instead")
                        src = "local CSV analysis (Snowflake failed)"
                if evidence is None:
                    if src == "Snowflake":
                        src = "local CSV analysis"
                    evidence = local_evidence(folder)
                evidence_src = src
                open(os.path.join(folder, "analysis.txt"), "w", encoding="utf-8").write(evidence)
                log("step 3: asking Gemini for proposals ...")
                raw = llm(build_prompt(base_settings, evidence, history, a.max_proposals))
                open(os.path.join(folder, "gemini_reply.json"), "w", encoding="utf-8").write(raw)
            try:
                items = parse_proposals(raw)
            except Exception as exc:
                log(f"could not read Gemini's reply as proposals ({exc}); stopping.")
                break

            # ---- guardrails -----------------------------------------------
            seen = {(r["setting"], r["new_value"]) for r in history if r.get("outcome") not in ("", "n/a")}
            logged_refusals = {(r["setting"], r["new_value"], r["decision"]) for r in history if r.get("outcome") == "n/a"}
            todo = []
            for item in items[:max(a.max_proposals * 2, 6)]:
                name = str(item.get("setting", "")) if isinstance(item, dict) else ""
                reason = str(item.get("reason", ""))[:200] if isinstance(item, dict) else ""
                row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
                       "iteration": it, "setting": name or "?", "old_value": fmt_val(base_settings.get(name, "?")),
                       "new_value": fmt_val(item.get("value", "?")) if isinstance(item, dict) else "?", "reason": reason,
                       "stage_reached": "none", "outcome": "n/a", "min_effect": a.min_effect,
                       "evidence_source": src, "agent_file": a.agent, "deals_per_match": a.deals}
                try:
                    value = validate(name, item.get("value") if isinstance(item, dict) else None)
                    row["new_value"] = fmt_val(value)
                    if value == base_settings[name]:
                        raise ValueError("same as the current value")
                    if (name, row["new_value"]) in seen and not a.retest:
                        raise ValueError("already tested in an earlier run (use --retest)")
                except ValueError as exc:
                    row["decision"] = f"not tested (guardrail: {exc})"
                    log(f"  refused: {name or '?'} {row['new_value']} -> {exc}")
                    key = (row["setting"], row["new_value"], row["decision"])
                    if "same as the current" not in str(exc) and key not in logged_refusals:
                        logged_refusals.add(key)
                        append_history(csv_path, row); new_rows.append(row)
                    continue
                if len(todo) < a.max_proposals:
                    todo.append((name, value, row))
            if not todo:
                log("no valid new proposals this round; stopping.")
                break

            # ---- test each change alone -----------------------------------
            base_mod = H.load_agent(a.agent, f"at_base_{it}")
            for k, v in accepted.items():
                setattr(base_mod, k, v)
            cache1, cache2, adopted_now = {}, {}, False
            for n, (name, value, row) in enumerate(todo, 1):
                var_mod = H.load_agent(a.agent, f"at_var_{it}_{n}")
                for k, v in {**accepted, name: value}.items():
                    setattr(var_mod, k, v)
                log(f"  testing {name} {row['old_value']} -> {row['new_value']}  ({row['reason'][:90]})")
                s1 = paired_stage(base_mod, var_mod, opps, a.deals, a.screen_seeds, f"{seed0}-i{it}-screen", cache1, "screen ")
                row.update(stage_reached="screen", screen_diff=round(s1["d"], 4), screen_lo=round(s1["lo"], 4),
                           screen_hi=round(s1["hi"], 4), screen_matches=s1["matches"], by_opponent=s1["by_opp"],
                           outcome=classify(s1["d"], a.min_effect))
                if s1["d"] < a.min_effect:
                    row["decision"] = "not adopted (failed screen)"
                else:
                    log("    passed the screen; confirming on fresh seeds ...")
                    s2 = paired_stage(base_mod, var_mod, opps, a.deals, a.confirm_seeds, f"{seed0}-i{it}-confirm", cache2, "confirm")
                    row.update(stage_reached="confirm", confirm_diff=round(s2["d"], 4), confirm_lo=round(s2["lo"], 4),
                               confirm_hi=round(s2["hi"], 4), confirm_matches=s2["matches"], by_opponent=s2["by_opp"],
                               outcome=classify(s2["d"], a.min_effect))
                    if s2["d"] >= a.min_effect and s2["lo"] > 0 and s2["loss_opp"]:
                        row["decision"] = f"not adopted (significantly worse vs {s2['loss_opp']})"
                    elif s2["d"] >= a.min_effect and s2["lo"] > 0:
                        row["decision"] = "ADOPTED (written to candidate file)"
                        accepted[name] = value
                        adopted_now = True
                    else:
                        if s2["d"] <= -a.min_effect:
                            why = "worse on fresh seeds"
                        elif s2["d"] >= a.min_effect:
                            why = "unconfirmed: interval includes 0"
                        else:
                            why = "gain not reproduced"
                        row["decision"] = f"not adopted ({why})"
                append_history(csv_path, row); new_rows.append(row)
                write_report(load_history(csv_path), md_path, a.min_effect)
                log(f"  -> {row['outcome']} | {row['decision']}")
                if adopted_now:
                    log("  a change was adopted, so the base changed: remaining proposals are re-asked next round.")
                    break
    except KeyboardInterrupt:
        log("interrupted: saving what was tested so far.")
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        log(f"STOPPED: {exc}")

    # ---- wrap-up ----------------------------------------------------------
    rows = load_history(csv_path)
    table = write_report(rows, md_path, a.min_effect) if rows else "(nothing was tested)"
    print("\n" + "=" * 70 + "\nREADME-ready table (full history in " + md_path + "):\n" + "=" * 70)
    print(table)
    if accepted:
        cand, diff = build_candidate(a.agent, accepted)
        score, secs = verify_candidate(cand)
        print(f"\nAdopted {len(accepted)} change(s). Candidate written to {cand}; smoke test OK "
              f"({score:+.2f} bb/deal vs always_pot over 100 deals, {secs:.0f}s, no illegal actions).")
        print("Diff:\n" + diff)
        if a.apply:
            apply_candidate(a.agent, cand)
        else:
            print(f"Your {a.agent} was NOT changed. Review the diff, then run:\n  copy {cand} {a.agent}   (or re-run with --apply)\n"
                  "and run: python -m pytest -q  plus a full judge match before submitting.")
    else:
        print(f"\nNo change was adopted, so {a.agent} stays as it is. That is a valid result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
