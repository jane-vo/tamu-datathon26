"""
gemini_helper.py - use Google AI Studio (Gemini) as an OFFLINE helper. Your bot never calls it.

Setup:
    pip install -r requirements-analysis.txt
    $env:GEMINI_API_KEY="your key"            (PowerShell;  Git Bash: export GEMINI_API_KEY=...)
    $env:GEMINI_MODEL="..."                   optional: use a model name that AI Studio lists for your key

Two jobs:
  1) analyze   - give it query results from Snowflake and get concrete tuning suggestions
       python gemini_helper.py analyze results.txt
  2) sparring  - have it write an opponent bot in a given style, saved as a ready-to-run bot file
       python gemini_helper.py sparring --name bluffer --style "raises weak hands a lot but folds to big re-raises"
     Then test against it:
       python harness_log.py --opponents sparring/gemini_bluffer.py --deals 300
     ALWAYS read the generated file before running it: it is code written by a model.

NEVER commit your API key. Do not paste it into chats or files.
"""
import argparse, datetime, os, re, sys

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

GAME = """Game: heads-up Short Deck (36 cards, 6..A) 5-card Pot-Limit Omaha, 80-chip stacks (40 big blinds, big blind = 2),
reset every hand. Hand = exactly 2 hole cards + exactly 3 board cards. Short-deck ranking: flush beats full house,
three of a kind beats a straight. Actions: FOLD, CHECK_CALL, RAISE_HALF (half pot), RAISE_POT (pot-limit max); max 4 raises per street.
Matches are duplicate (same cards both seats), 2000 hands, scored by a 95% confidence interval of bb/deal."""

OBS = """obs fields: hand_index, street (Street.PREFLOP/FLOP/TURN/RIVER), hole_cards (5 ints 0..35, rank=c>>2 where 0=six..8=ace, suit=c&3),
board (0,3,4 or 5 ints), pot, to_call, my_bet, opp_bet, my_stack, opp_stack, is_button, legal_actions (tuple of Action),
raise_to (dict Action->int), min_raise_to, raises_this_street, action_history (entries with .street .is_me .action .amount .street_total), big_blind."""


def cmd_export(args):
    """Pull Query 2 results directly from Snowflake using environment variables."""
    try:
        import snowflake.connector
    except ImportError:
        sys.exit("snowflake-connector-python is not installed. Run: pip install snowflake-connector-python[pandas]")
    
    missing = [k for k in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD") if not os.environ.get(k)]
    if missing:
        sys.exit("Set these environment variables first: " + ", ".join(missing))

    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "POKER_BOT"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
        enable_server_session_keep_alive_auto_detection=True
    )

    query = """
    SELECT STREET, FLOOR(EQUITY * 10) / 10 AS EQUITY_BUCKET, ACTION,
           COUNT(*) AS N, ROUND(AVG(HAND_NET_CHIPS), 2) AS AVG_NET
    FROM DECISIONS
    WHERE EXPLORED = 1 AND EQUITY IS NOT NULL
    GROUP BY 1, 2, 3 HAVING COUNT(*) >= 30
    ORDER BY STREET, EQUITY_BUCKET, ACTION;
    """

    import pandas as pd
    df = pd.read_sql(query, conn)
    df.to_csv(args.out, index=False)
    conn.close()
    print(f"Successfully exported Query 2 results to {args.out} ({len(df)} rows).")


def ask(prompt):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (see the top of this file).")
    from google import genai
    client = genai.Client(api_key=key)
    return client.models.generate_content(model=MODEL, contents=prompt).text


def settings_block(agent_path="agent.py"):
    try:
        text = open(agent_path, encoding="utf-8").read()
        a, b = text.index("# Tunable settings"), text.index("# Preflop strength model")
        return text[a:b]
    except (OSError, ValueError):
        return "(could not read the tunable settings from agent.py)"


def cmd_analyze(args):
    data = "\n\n".join(open(p, encoding="utf-8").read() for p in args.files)
    prompt = f"""You are helping tune a poker bot for this competition.
{GAME}

The bot estimates win chance vs a random hand (formula preflop, Monte Carlo later streets), compares it to pot odds,
and uses these tunable settings:
{settings_block(args.agent)}

Here are query results from our own match logs (Snowflake). Rows with EXPLORED=1 had a random action, so comparing
actions among them is unbiased. Results are noisy; hand outcomes have a standard deviation of about 25 bb per deal.
{data[:30000]}

Task: (1) list the 3 most likely leaks, citing the numbers; (2) for each, propose ONE concrete change to a setting above,
with the new value; (3) say how many hands we would need to confirm each change. Be skeptical of small samples."""
    answer = ask(prompt)
    print(answer)
    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", f"gemini_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.md")
    open(out, "w", encoding="utf-8").write(answer)
    print(f"\n(saved to {out})")


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def build_bot(code, name, template="baseline-bots/baseline_random.py", out_dir="sparring"):
    """Wrap a generated decide(obs, memory) function into a complete, safe bot file."""
    src = open(template, encoding="utf-8").read()
    start_m, end_m = "# ===== YOUR CODE HERE =====", "# ===== END OF YOUR CODE ====="
    a, b = src.index(start_m), src.index(end_m) + len(end_m)
    block = f'''{start_m}

import random

# ---- generated by Gemini: READ IT BEFORE RUNNING ----
{code}
# ---- end of generated code ----


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = random.Random(7)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Run the generated strategy; fall back to a legal action if it fails or answers illegally."""
    try:
        action = decide(obs, memory)
        if action in obs.legal_actions:
            return action
    except Exception:
        pass
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass

{end_m}'''
    new = src[:a] + block + src[b:]
    new = re.sub(r'AGENT_NAME = ".*?"', f'AGENT_NAME = "Gemini-{name}"', new, count=1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"gemini_{name}.py")
    open(path, "w", encoding="utf-8").write(new)
    compile(new, path, "exec")  # syntax check
    return path


def cmd_sparring(args):
    prompt = f"""Write a poker opponent for testing purposes.
{GAME}
{OBS}
Style: {args.style}

Return ONLY one Python code block containing a function `def decide(obs, memory):` that returns an Action
(Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF or Action.RAISE_POT; `Action` and `Street` are already imported).
Rules: only use obs fields listed above; FOLD is only legal when obs.to_call > 0; raises only if in obs.legal_actions;
`memory` is a dict you may use for tracking, and memory["rng"] is a random.Random. You may `import math`. Keep it under 80 lines,
fast (no loops over thousands of items), and add short comments. Do not read files or use the network."""
    code = extract_code(ask(prompt))
    path = build_bot(code, args.name)
    print(f"Wrote {path}. Read it, then test:\n  python harness_log.py --opponents {path.replace(os.sep, '/')} --deals 300")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    # Export command
    e = sub.add_parser("export"); e.add_argument("--out", default="results.txt")
    e.set_defaults(fn=cmd_export)

    # Analyze command
    a = sub.add_parser("analyze"); a.add_argument("files", nargs="+"); a.add_agent = a.add_argument("--agent", default="agent.py")
    a.set_defaults(fn=cmd_analyze)

    # Sparring command
    s = sub.add_parser("sparring"); s.add_argument("--name", required=True); s.add_argument("--style", required=True)
    s.set_defaults(fn=cmd_sparring)

    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
