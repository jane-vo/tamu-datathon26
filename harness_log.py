"""
harness_log.py - play duplicate matches (no server needed) and LOG every decision your bot makes.

Needs harness.py, game.py, evaluator.py and agent.py in the same folder (the repo root).
Writes two CSV files into logs/ that snowflake_load.py can upload.

Examples (run from the repo root):
    python harness_log.py --opponents always_pot --deals 200 --eps 0.1 --version v1
    python harness_log.py --opponents random,calling_station,always_pot,always_half --seeds 3 --deals 1000
    python harness_log.py --opponents always_pot --version margin05 --param BASE_MARGIN=0.05
    python harness_log.py --opponents sparring/gemini_bluffer.py --version v1

--eps is the chance that a decision is replaced by a RANDOM legal action (marked EXPLORED=true).
That randomness is what lets you compare actions fairly in SQL later. Use ~0.1 for data
collection and 0 when you want a clean measurement of strength.
"""
import argparse, ast, csv, datetime, os, random, sys, time, uuid

import harness as H
from evaluator import cards_to_str

BASELINES = ("random", "calling_station", "always_pot", "always_half")
DECISION_COLS = ["RUN_ID", "BOT_VERSION", "OPPONENT", "HALF", "HAND_INDEX", "STEP", "IS_BUTTON", "STREET",
                 "HOLE", "BOARD", "EQUITY", "POT", "TO_CALL", "MY_STACK", "OPP_STACK", "RAISES_THIS_STREET",
                 "LEGAL", "ACTION", "EXPLORED", "HAND_NET_CHIPS", "WENT_TO_SHOWDOWN"]
MATCH_COLS = ["RUN_ID", "BOT_VERSION", "OPPONENT", "SEED", "DEALS", "EPS", "MEAN_BB_PER_DEAL", "CI_LOW",
              "CI_HIGH", "VERDICT", "MY_COMPUTE_SECONDS_PER_HALF", "PARAMS", "CREATED_AT"]


def load_bot(spec, name):
    """spec is a baseline name (e.g. always_pot) or a path to any bot .py file."""
    path = spec if spec.endswith(".py") else os.path.join("baseline-bots", f"baseline_{spec}.py")
    return H.load_agent(path, name)


class LoggedBot(H.ModBot):
    """Your bot, plus a log row for every decision (filled with the hand's result at hand end)."""

    def __init__(self, mod, sink, run_id, version, opponent, eps, eq_samples, seed):
        super().__init__(mod)
        self.sink, self.run_id, self.version, self.opponent = sink, run_id, version, opponent
        self.eps, self.eq_samples = eps, eq_samples
        self.rng = random.Random(f"explore-{seed}")       # exploration choices
        self.eq_rng = random.Random(f"equity-{seed}")     # equity estimates for the log only
        self.half = 0
        self.rows, self.step, self.cur_hand = [], 0, None

    def start(self):
        super().start()
        self.half += 1

    def _equity(self, obs):
        try:
            if len(obs.board) == 0:
                return round(self.m.preflop_equity(obs.hole_cards), 4)
            return round(self.m.mc_equity(obs.hole_cards, obs.board, self.eq_samples, self.eq_rng), 4)
        except Exception:
            return ""

    def act(self, obs):
        action = super().act(obs)
        explored = False
        if self.eps > 0 and self.rng.random() < self.eps:
            action = self.rng.choice(list(obs.legal_actions))
            explored = True
        if obs.hand_index != self.cur_hand:
            self.cur_hand, self.step = obs.hand_index, 0
        self.step += 1
        self.rows.append({
            "RUN_ID": self.run_id, "BOT_VERSION": self.version, "OPPONENT": self.opponent, "HALF": self.half,
            "HAND_INDEX": obs.hand_index, "STEP": self.step, "IS_BUTTON": int(obs.is_button),
            "STREET": obs.street.name, "HOLE": cards_to_str(obs.hole_cards), "BOARD": cards_to_str(obs.board),
            "EQUITY": self._equity(obs), "POT": obs.pot, "TO_CALL": obs.to_call, "MY_STACK": obs.my_stack,
            "OPP_STACK": obs.opp_stack, "RAISES_THIS_STREET": obs.raises_this_street,
            "LEGAL": "|".join(a.name for a in obs.legal_actions), "ACTION": action.name,
            "EXPLORED": int(explored), "HAND_NET_CHIPS": "", "WENT_TO_SHOWDOWN": ""})
        return action

    def end(self, result):
        super().end(result)
        for row in self.rows:
            row["HAND_NET_CHIPS"] = result.my_net
            row["WENT_TO_SHOWDOWN"] = int(result.went_to_showdown)
            self.sink.writerow(row)
        self.rows = []


def apply_params(mod, items):
    """--param NAME=VALUE overrides a constant in agent.py (value is a Python literal)."""
    shown = []
    for item in items or []:
        name, _, raw = item.partition("=")
        if not hasattr(mod, name):
            sys.exit(f"agent.py has no setting called {name!r}")
        setattr(mod, name, ast.literal_eval(raw))
        shown.append(item)
    return ";".join(shown)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="agent.py", help="your bot file (default agent.py)")
    p.add_argument("--opponents", default="always_pot", help="comma list: baseline names or paths to .py bots")
    p.add_argument("--deals", type=int, default=200, help="deals per half (default 200; the real match is 1000)")
    p.add_argument("--seeds", type=int, default=1, help="how many different card seeds per opponent")
    p.add_argument("--seed", default="1", help="base seed text")
    p.add_argument("--eps", type=float, default=0.0, help="chance of a random exploratory action per decision")
    p.add_argument("--version", default="v1", help="label stored with every row (e.g. v1, margin05)")
    p.add_argument("--param", action="append", help="override a constant, e.g. --param BASE_MARGIN=0.05")
    p.add_argument("--equity-samples", type=int, default=150, help="simulations for the logged equity (postflop)")
    p.add_argument("--out", default="logs", help="output folder")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dec_path = os.path.join(a.out, f"decisions_{stamp}.csv")
    mat_path = os.path.join(a.out, f"matches_{stamp}.csv")
    me = H.load_agent(a.agent, "me_logged")
    params = apply_params(me, a.param)

    with open(dec_path, "w", newline="") as df, open(mat_path, "w", newline="") as mf:
        dec = csv.DictWriter(df, fieldnames=DECISION_COLS)
        mat = csv.DictWriter(mf, fieldnames=MATCH_COLS)
        dec.writeheader(); mat.writeheader()
        for opp_spec in [s.strip() for s in a.opponents.split(",") if s.strip()]:
            opp_mod = load_bot(opp_spec, "opp_" + os.path.basename(opp_spec).replace(".", "_"))
            opp_name = os.path.splitext(os.path.basename(opp_spec))[0].replace("baseline_", "")
            for k in range(a.seeds):
                seed = f"{a.seed}-{opp_name}-{k}"
                run_id = uuid.uuid4().hex[:12]
                bot = LoggedBot(me, dec, run_id, a.version, opp_name, a.eps, a.equity_samples, seed)
                t0 = time.time()
                mu, lo, hi, verdict, comp = H.match(bot, H.ModBot(opp_mod), a.deals, seed=seed)
                mat.writerow({"RUN_ID": run_id, "BOT_VERSION": a.version, "OPPONENT": opp_name, "SEED": seed,
                              "DEALS": a.deals, "EPS": a.eps, "MEAN_BB_PER_DEAL": round(mu, 4),
                              "CI_LOW": round(lo, 4), "CI_HIGH": round(hi, 4), "VERDICT": verdict,
                              "MY_COMPUTE_SECONDS_PER_HALF": round(comp, 2), "PARAMS": params,
                              "CREATED_AT": datetime.datetime.now().isoformat(timespec="seconds")})
                df.flush(); mf.flush()
                print(f"{a.version:10s} vs {opp_name:16s} seed {seed:24s} {mu:+7.3f} bb/deal "
                      f"[{lo:+.3f},{hi:+.3f}] {verdict:4s} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nWrote {dec_path}\n      {mat_path}")


if __name__ == "__main__":
    main()
