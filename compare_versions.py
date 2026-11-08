"""
compare_versions.py - paired (duplicate) comparison helpers used by autotune.py.

Two bot versions that play the same opponent with the same seed get the same cards, so the
per-deal difference between them removes most of the card luck.

    deal_results(bot_mod, opp_mod, deals, seed) -> list of bb results, one per deal (both seats)
    mean_se(xs)                                 -> (mean, standard error)
    load_opp(spec)                              -> (name, module); spec is a baseline name or a .py path

Command line (quick A/B of two agent files):
    python compare_versions.py agent.py agent_candidate.py --opponents always_pot,random --deals 500
"""
import argparse, math, os, sys

import harness as H

_loaded = {}


def load_opp(spec):
    """Load an opponent once. Returns (short name, module)."""
    path = spec if spec.endswith(".py") else os.path.join("baseline-bots", f"baseline_{spec}.py")
    name = os.path.splitext(os.path.basename(path))[0].replace("baseline_", "")
    if path not in _loaded:
        if not os.path.exists(path):
            raise FileNotFoundError(f"opponent file missing: {path}")
        mod_name = "cv_opp_" + "".join(ch if ch.isalnum() else "_" for ch in path)
        _loaded[path] = H.load_agent(path, mod_name)
    return name, _loaded[path]


def deal_results(bot_mod, opp_mod, deals, seed):
    """Duplicate match: play `deals` deals in each seat with the same cards.
    Returns one value per deal in big blinds (same convention as harness.match)."""
    me, opp = H.ModBot(bot_mod), H.ModBot(opp_mod)
    h1 = H.play_half(me, opp, deals, seed, 0)
    h2 = H.play_half(me, opp, deals, seed, 1)
    return [(x + y) / 2.0 for x, y in zip(h1, h2)]


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu, sd / math.sqrt(n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base"); p.add_argument("variant")
    p.add_argument("--opponents", default="random,calling_station,always_pot,always_half")
    p.add_argument("--deals", type=int, default=500)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", default="cmp")
    a = p.parse_args()
    base, var = H.load_agent(a.base, "cv_base"), H.load_agent(a.variant, "cv_var")
    diffs = []
    for spec in [s.strip() for s in a.opponents.split(",") if s.strip()]:
        name, opp = load_opp(spec)
        d = []
        for k in range(a.seeds):
            seed = f"{a.seed}-{name}-{k}"
            d += [v - b for v, b in zip(deal_results(var, opp, a.deals, seed), deal_results(base, opp, a.deals, seed))]
        mu, se = mean_se(d)
        print(f"vs {name:16s} {mu:+.3f} bb/deal [{mu - 1.96 * se:+.3f}, {mu + 1.96 * se:+.3f}]", flush=True)
        diffs += d
    mu, se = mean_se(diffs)
    print(f"overall          {mu:+.3f} bb/deal [{mu - 1.96 * se:+.3f}, {mu + 1.96 * se:+.3f}]")


if __name__ == "__main__":
    sys.exit(main())
