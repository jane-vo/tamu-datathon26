"""
snowflake_load.py - upload the CSVs written by harness_log.py to Snowflake.

Set these environment variables first (PowerShell example in the comments):
    SNOWFLAKE_ACCOUNT     e.g. myorg-myaccount      ($env:SNOWFLAKE_ACCOUNT="myorg-myaccount")
    SNOWFLAKE_USER
    SNOWFLAKE_PASSWORD
    SNOWFLAKE_WAREHOUSE   optional, default COMPUTE_WH
    SNOWFLAKE_DATABASE    optional, default POKER_BOT
    SNOWFLAKE_SCHEMA      optional, default PUBLIC
    SNOWFLAKE_ROLE        optional

Usage:
    python snowflake_load.py --check        # just test the connection
    python snowflake_load.py                # load every new file in logs/ (then move it to logs/loaded/)
    python snowflake_load.py --dry-run      # show which files would be loaded

NEVER put these values in a file you commit: the repo is public.
"""

"""
snowflake_load.py - upload the CSVs written by harness_log.py to Snowflake.
...
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse, glob, os, shutil, sys

import envfile
envfile.load()                     # SNOWFLAKE_* from .env (shell variables win)

DDL = {
    "DECISIONS": """CREATE TABLE IF NOT EXISTS DECISIONS (
        RUN_ID STRING, BOT_VERSION STRING, OPPONENT STRING, HALF INT, HAND_INDEX INT, STEP INT,
        IS_BUTTON INT, STREET STRING, HOLE STRING, BOARD STRING, EQUITY FLOAT, POT INT, TO_CALL INT,
        MY_STACK INT, OPP_STACK INT, RAISES_THIS_STREET INT, LEGAL STRING, ACTION STRING,
        EXPLORED INT, HAND_NET_CHIPS INT, WENT_TO_SHOWDOWN INT)""",
    "MATCHES": """CREATE TABLE IF NOT EXISTS MATCHES (
        RUN_ID STRING, BOT_VERSION STRING, OPPONENT STRING, SEED STRING, DEALS INT, EPS FLOAT,
        MEAN_BB_PER_DEAL FLOAT, CI_LOW FLOAT, CI_HIGH FLOAT, VERDICT STRING,
        MY_COMPUTE_SECONDS_PER_HALF FLOAT, PARAMS STRING, CREATED_AT STRING)""",
}


def connect():
    import snowflake.connector
    missing = [k for k in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD") if not os.environ.get(k)]
    if missing:
        sys.exit("Set these environment variables first: " + ", ".join(missing))
    
    kw = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"], 
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        enable_server_session_keep_alive_auto_detection=True  # <--- Added this line
    )
    
    if os.environ.get("SNOWFLAKE_ROLE"):
        kw["role"] = os.environ["SNOWFLAKE_ROLE"]
        
    return snowflake.connector.connect(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only test the connection")
    ap.add_argument("--dry-run", action="store_true", help="list files, upload nothing")
    ap.add_argument("--dir", default="logs", help="folder with the CSVs (default logs)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "decisions_*.csv")) +
                   glob.glob(os.path.join(args.dir, "matches_*.csv")))
    if args.dry_run:
        print("Would load:", *files, sep="\n  ") if files else print("No new files in", args.dir)
        return

    conn = connect()
    cur = conn.cursor()
    db = os.environ.get("SNOWFLAKE_DATABASE", "POKER_BOT")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    cur.execute("SELECT CURRENT_VERSION()")
    print("Connected. Snowflake version:", cur.fetchone()[0])
    if args.check:
        return
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cur.execute(f"USE DATABASE {db}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cur.execute(f"USE SCHEMA {schema}")
    for ddl in DDL.values():
        cur.execute(ddl)

    import pandas as pd
    from snowflake.connector.pandas_tools import write_pandas
    done_dir = os.path.join(args.dir, "loaded")
    os.makedirs(done_dir, exist_ok=True)
    for path in files:
        table = "DECISIONS" if os.path.basename(path).startswith("decisions_") else "MATCHES"
        df = pd.read_csv(path)
        df.columns = [c.upper() for c in df.columns]
        ok, _, nrows, _ = write_pandas(conn, df, table, database=db, schema=schema)
        print(f"{os.path.basename(path)} -> {db}.{schema}.{table}: {nrows} rows, success={ok}")
        if ok:
            shutil.move(path, os.path.join(done_dir, os.path.basename(path)))  # never load a file twice
    conn.close()


if __name__ == "__main__":
    main()
