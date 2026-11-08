"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

How it plays (v4):
    * numpy Monte Carlo (~20x faster than pure Python), so every decision uses ~1000 samples.
    * A particle filter over the opponent's possible hands: at the start of a hand we sample
      opponent hands, and every opponent action re-weights them by how likely THIS opponent
      (its measured bet / call / fold frequencies) is to take that action with that hand.
    * Each decision compares the expected value of fold / call / half-pot / pot, using the
      opponent's measured fold rate and our equity against the hands that would continue.
    * If numpy is missing, it falls back to the v3 threshold bot (pure Python).
"""

import os
import random
import time
import traceback
from itertools import combinations

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, get_table, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "GOOFY_GAMBLERS"
AGENT_NAME = "GoofyPokerBot"


# ===== YOUR CODE HERE =====

# ---------------------------------------------------------------------------
# Tunable settings
# ---------------------------------------------------------------------------
# The judge charges our compute PLUS ~0.65-0.8 ms of HTTP/JSON work per request (measured
# locally), so the compute budget is what is left of BANK_TARGET_S after the requests this
# half is projected to need: ~3 s vs bots that fold a lot, less vs calling stations.
BANK_TARGET_S = 6.2       # aim for this much of the judge's 8 s bank per half
OVERHEAD_MS = 0.9         # judge-charged time per request outside our code (conservative)
BANK_BUDGET_S = 3.0       # most compute we ever plan for per half (least: 1.5 s)
BANK_SAFE_S = 6.9         # if our estimate of the judge's clock passes this, switch to zero-compute play
MAX_DECISION_MS = 25.0    # never spend more than this on one decision (hard limit is 1000 ms)
MIN_SAMPLES = 10          # (v3 fallback only)
MAX_SAMPLES = 400         # (v3 fallback only)
POOL_MIN = 120            # opponent hands sampled per hand (numpy engine), steered between these
POOL_MAX = 1600

# Decision model (numpy engine)
CALL_MARGIN = 0.02        # equity tax on calling when more streets are to come (out of position: 2x)
RAISE_EDGE = 0.01         # postflop: a raise must beat the best passive action by this share of the pot
RAISE_EDGE_PRE = 0.60     # preflop: same, much higher (keep pots small and use our postflop edge)
FOLD_PRIOR = 0.25         # prior: opponent folds this often when facing a bet
BET_PRIOR = 0.35          # prior: opponent bets this often when nothing to call
RANGE_TAU = 0.15          # softness of the range model (bigger = trust opponent actions less)
BLUFF_FLOOR = 0.20        # share of an opponent's bets we assume can be any hand
SIZE_FOLD = 0.08          # pot-size raises fold this much more often than half-pot ones (relative)

# v3 thresholds (used only by the pure-Python fallback)
OPEN_PRE = (0.545, 0.60)      # preflop, nothing to call
OPEN_POST = (0.60, 0.75)      # postflop, nothing to call
RERAISE_PRE = (0.58, 0.64)    # facing a bet, preflop
RERAISE_POST = (0.68, 0.82)   # facing a bet, postflop
BASE_MARGIN = 0.03            # extra equity needed on top of pot odds to call
BLUFF_RATE = 0.20             # chance to bet a weak hand when the opponent folds too much

# Preflop strength model, fitted offline (fit_preflop.py) to Monte Carlo equity.
# Features: see _preflop_features().
PREFLOP_COEF = [0.42489, 0.0355, -0.07483, -0.24388, 0.02738, -0.00945,
                -0.00425, 0.00034, 0.00038, -0.01571, 0.37711, -0.07651]

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))
_WINDOWS = ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [8, 0, 1, 2, 3])
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def _preflop_features(hole):
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    cnt = [0] * 9
    for r in ranks:
        cnt[r] += 1
    pair_hi = sum(1 + r / 8 for r in range(9) if cnt[r] == 2)
    trips = sum(1 for r in range(9) if cnt[r] == 3)
    quads = sum(1 for r in range(9) if cnt[r] >= 4)
    sc = [0] * 4
    for s in suits:
        sc[s] += 1
    ace_suited = king_suited = 0
    for c in hole:
        if sc[c & 3] >= 2:
            if c >> 2 == 8:
                ace_suited = 1
            elif c >> 2 == 7:
                king_suited = 1
    conn = max(sum(1 for r in w if cnt[r]) for w in _WINDOWS)
    return [1.0, pair_hi, trips, quads, sum(n * (n - 1) // 2 for n in sc), ace_suited,
            king_suited, conn, sum(1 for r in range(9) if cnt[r]),
            cnt[8] + cnt[7] + cnt[6], sum(ranks) / 40.0, max(sc)]


def preflop_equity(hole) -> float:
    """Estimated win chance of 5 hole cards vs a random hand (no simulation, ~10 microseconds)."""
    e = sum(c * f for c, f in zip(PREFLOP_COEF, _preflop_features(hole)))
    return min(0.75, max(0.28, e))


def mc_equity(hole, board, n, rng) -> float:
    """Monte Carlo win chance vs a random opponent hand (pure Python; used by the fallback
    and by harness_log.py for logging)."""
    table = get_table()
    dead = set(hole) | set(board)
    rem = [c for c in range(36) if c not in dead]
    need = 5 - len(board)
    hb = [1 << c for c in hole]
    hm = [hb[i] | hb[j] for i, j in _PAIRS]
    bb0 = [1 << c for c in board]
    sample = rng.sample
    wins = 0.0
    if need == 0:  # river: our score and the board masks never change
        bm = [bb0[i] | bb0[j] | bb0[k] for i, j, k in _TRIPLES]
        mine = max([table[h | b] for h in hm for b in bm])
        for _ in range(n):
            ob = [1 << c for c in sample(rem, 5)]
            om = [ob[i] | ob[j] for i, j in _PAIRS]
            opp = max([table[h | b] for h in om for b in bm])
            wins += 1.0 if mine > opp else (0.5 if mine == opp else 0.0)
        return wins / n
    for _ in range(n):
        d = sample(rem, need + 5)
        bb = bb0 + [1 << c for c in d[:need]]
        bm = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
        ob = [1 << c for c in d[need:]]
        om = [ob[i] | ob[j] for i, j in _PAIRS]
        mine = max([table[h | b] for h in hm for b in bm])
        opp = max([table[h | b] for h in om for b in bm])
        wins += 1.0 if mine > opp else (0.5 if mine == opp else 0.0)
    return wins / n


# ---------------------------------------------------------------------------
# numpy engine: vectorised Omaha scoring
#   a 2-card hole pair and a 3-card board triple each map to an index (rank multiset x suit
#   code); one table lookup scores the 5-card hand. 100 lookups per player per sample.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    HAVE_NP = True
except Exception:  # pragma: no cover - the tournament image ships numpy
    np = None
    HAVE_NP = False

if HAVE_NP:
    def _build_engine():
        from evaluator import _score_ranks
        pairs = sorted({(a, b) for a in range(9) for b in range(a, 9)})          # 45 rank pairs
        trips = sorted({(a, b, c) for a in range(9) for b in range(a, 9) for c in range(b, 9)})  # 165
        pidx = {p: i for i, p in enumerate(pairs)}
        tidx = {t: i for i, t in enumerate(trips)}
        nt = len(trips) * 5
        raw = {}
        for p, i in pidx.items():
            for t, j in tidx.items():
                rs = p + t
                if max(rs.count(r) for r in rs) > 4:
                    continue
                desc = tuple(sorted(rs, reverse=True))
                plain = _score_ranks(desc, False)
                flush = _score_ranks(desc, True) if len(set(rs)) == 5 else plain
                for ps in range(5):
                    for ts in range(5):
                        is_fl = ps > 0 and ps == ts
                        raw[(i * 5 + ps) * nt + j * 5 + ts] = flush if is_fl else plain
        dense = {s: k + 1 for k, s in enumerate(sorted(set(raw.values())))}
        tbl = np.zeros(len(pairs) * 5 * nt, np.int16)
        for k, s in raw.items():
            tbl[k] = dense[s]
        pair_key = np.zeros((36, 36), np.int32)
        for a in range(36):
            for b in range(36):
                ra, rb = sorted((a >> 2, b >> 2))
                sc = (a & 3) + 1 if (a & 3) == (b & 3) else 0
                pair_key[a, b] = (pidx[(ra, rb)] * 5 + sc) * nt
        trip_key = np.zeros((36, 36, 36), np.int32)
        for a in range(36):
            for b in range(36):
                for c in range(36):
                    r = tuple(sorted((a >> 2, b >> 2, c >> 2)))
                    sc = (a & 3) + 1 if (a & 3) == (b & 3) == (c & 3) else 0
                    trip_key[a, b, c] = tidx[r] * 5 + sc
        return tbl, pair_key, trip_key

    _TBL, _PK, _TK = _build_engine()
    _NT = 165 * 5                                          # number of distinct triple keys
    _P0, _P1 = (np.array(x) for x in zip(*_PAIRS))
    _T0, _T1, _T2 = (np.array(x) for x in zip(*_TRIPLES))
    _BANK = np.argsort(np.random.default_rng(7).random((24000, 36)), axis=1).astype(np.int8)
    _AR9, _AR4 = np.arange(9), np.arange(4)
    _WIN = np.array(_WINDOWS)
    _COEF = np.array(PREFLOP_COEF)
    _ALL_TK = np.arange(_NT)
    _AR = np.arange(4096)
    _ROWS = _AR[:, None]

    def _pair_keys(h):            # (..., 5) hole cards -> (..., 10)
        return _PK[h[..., _P0], h[..., _P1]]

    def _trip_keys(b):            # (..., 5) board -> (..., 10)
        return _TK[b[..., _T0], b[..., _T1], b[..., _T2]]

    def _partial_trip_keys(board):  # 3 or 4 board cards -> (#triples,)
        b = list(board)
        return np.array([_TK[b[i], b[j], b[k]] for i, j, k in combinations(range(len(b)), 3)])

    def _best(pk, tk):            # broadcast pk (...,10) with tk (...,T) -> best score (...)
        s = _TBL[pk[..., :, None] + tk[..., None, :]]
        return s.reshape(s.shape[:-2] + (-1,)).max(axis=-1)

    def _np_preflop(h):           # vectorised preflop_equity (same fitted formula)
        r, s = h >> 2, h & 3
        cnt = (r[:, :, None] == _AR9).sum(1)
        sc = (s[:, :, None] == _AR4).sum(1)
        scc = np.take_along_axis(sc, s, 1) >= 2
        present = cnt > 0
        f = np.stack([np.ones(len(h)), ((cnt == 2) * (1 + _AR9 / 8)).sum(1), (cnt == 3).sum(1),
                      (cnt >= 4).sum(1), (sc * (sc - 1) // 2).sum(1), ((r == 8) & scc).any(1),
                      ((r == 7) & scc).any(1), present[:, _WIN].sum(2).max(1), present.sum(1),
                      cnt[:, 6:].sum(1), r.sum(1) / 40.0, sc.max(1)], 1)
        return f @ _COEF

    def _draw(alive, need, rng):
        """For each row of `alive` (N,36 bool), `need` distinct random alive cards -> (N, need)."""
        n = alive.shape[0]
        off = int(rng.integers(0, len(_BANK) - n)) if len(_BANK) > n else 0
        perm = _BANK[off:off + n]
        a = alive[_ROWS[:n], perm]
        sel = a & (np.cumsum(a, axis=1) <= need)
        return perm[sel].reshape(n, need).astype(np.intp)

    def _draw_few(mask, rem, need, rng):
        """need = 1 or 2 run-out cards per row from `rem`, never one of that row's opponent
        cards (mask). Exact and loop-free: each row has (at least) len(rem) - 5 live cards, so
        pick uniform positions among them and map positions to cards with a cumulative count."""
        n, r = mask.shape[0], len(rem)
        live = np.cumsum(~mask[:, rem], axis=1)                 # live cards up to each column
        j = rng.integers(0, r - 5, n)
        out = np.empty((n, need), np.intp)
        out[:, 0] = rem[(live > j[:, None]).argmax(1)]
        if need == 2:
            k = rng.integers(0, r - 6, n)
            k += k >= j                                         # a different position, uniform
            out[:, 1] = rem[(live > k[:, None]).argmax(1)]
        return out

    def _pct(x, valid):
        """Percentile (0..1) of each x among the valid rows (ties get the middle)."""
        xs = np.sort(x[valid])
        if len(xs) == 0:
            return np.full(len(x), 0.5)
        return (np.searchsorted(xs, x, "left") + np.searchsorted(xs, x, "right")) / (2.0 * len(xs))

    def _sig(z):
        return 0.5 + 0.5 * np.tanh(0.5 * z)       # logistic function, no overflow

    def _my_table(hole):
        """Our best score for every possible board triple (825,), so scoring our hand on a
        5-card board is 10 lookups instead of 100."""
        mpk = _pair_keys(np.array(hole))
        return _TBL[mpk[:, None] + _ALL_TK[None, :]].max(0)

    def _build_opp_bank(k=40000):
        """Random opponent hands with their pair keys and preflop-strength percentile."""
        g = np.random.default_rng(11)
        hands = np.argsort(g.random((k, 36)), axis=1)[:, :5].astype(np.intp)
        mask = np.zeros((k, 36), bool)
        np.put_along_axis(mask, hands, True, 1)
        pre = _np_preflop(hands)
        pre_pct = _pct(pre, np.ones(k, bool))
        return hands, mask, _pair_keys(hands), pre_pct

    _OB_HANDS, _OB_MASK, _OB_PK, _OB_PRE = _build_opp_bank()


# ---------------------------------------------------------------------------
# Memory / opponent model
# ---------------------------------------------------------------------------
def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands). Set up everything we track."""
    memory["hands_played"] = 0
    memory["rng"] = random.Random(2026)
    memory["np_rng"] = np.random.default_rng(2026) if HAVE_NP else None
    memory["t_used"] = 0.0            # our own compute time this half (seconds)
    memory["calls"] = 0               # decisions this half (for the judge-clock estimate)
    memory["us_per_sample"] = 60.0    # (fallback) running estimate of one Monte Carlo sample
    memory["pool_n"] = 500.0          # (numpy) opponent hands per pool, steered by _govern()
    memory["np_hand_s"] = 0.002       # running cost of a hand that used the pool (seconds)
    memory["np_frac"] = 0.7           # running share of hands that used the pool
    memory["num_hands"] = config.num_hands
    # opponent frequencies: [preflop, postflop]
    memory["o_open"] = [[0, 0], [0, 0]]       # nothing to call: [opportunities, bets]
    # facing a bet: [folds, calls, raises] per bucket = 2 * (postflop) + (they already raised this street)
    memory["o_face"] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # v3 fallback statistics
    memory["opp_actions"] = 0
    memory["opp_raises"] = 0
    memory["i_raised_hand"] = False
    memory["my_raise_hands"] = 0
    memory["opp_folds_to_me"] = 0
    memory["seen_hand"] = -1
    memory["seen_actions"] = 0
    memory["hand"] = None             # per-hand state (history scan + particle filter)


def _bet_freq(memory, g):
    n, b = memory["o_open"][g]
    return (b + 6 * BET_PRIOR) / (n + 6)


def _face_freqs(memory, b):
    """(fold, call, raise) frequencies of the opponent when facing a bet, for bucket b.
    Odd buckets = they had already raised this street (they fold much less to a re-raise)."""
    f, c, r = memory["o_face"][b]
    pf, pr = (FOLD_PRIOR, 0.15) if b % 2 == 0 else (FOLD_PRIOR * 0.5, 0.2)
    n = f + c + r + 8
    return (f + 8 * pf) / n, (c + 8 * (1 - pf - pr)) / n, (r + 8 * pr) / n


def _new_hand_state(obs):
    mine, theirs = (1, 2) if obs.is_button else (2, 1)   # blinds: the button posts the small blind
    return {"idx": obs.hand_index, "n": 0, "street": Street.PREFLOP, "mine": mine, "theirs": theirs,
            "opp_acted_street": False, "opp_raised": False, "pending": None, "events": [], "pf": None}


def _scan_history(obs, memory):
    """Read new history entries once: update opponent frequencies and queue opponent
    actions (with the situation they faced) for the particle filter."""
    hs = memory["hand"]
    if hs is None or hs["idx"] != obs.hand_index:
        hs = memory["hand"] = _new_hand_state(obs)
    for a in obs.action_history[hs["n"]:]:
        if a.street != hs["street"]:
            hs["street"], hs["mine"], hs["theirs"] = a.street, 0, 0
            hs["opp_acted_street"] = hs["opp_raised"] = False
        g = 0 if a.street == Street.PREFLOP else 1
        if a.is_me:
            hs["mine"] = a.street_total
        else:
            hs["pending"] = None
            facing = hs["mine"] - hs["theirs"]
            b = 2 * g + int(hs["opp_raised"])
            if facing > 0:
                k = 0 if a.action == Action.FOLD else (2 if a.action in _RAISES else 1)
                memory["o_face"][b][k] += 1
                kind = ("face_fold", "face_call", "face_raise")[k]
            else:
                memory["o_open"][g][0] += 1
                bet = a.action in _RAISES
                memory["o_open"][g][1] += int(bet)
                kind = "open_bet" if bet else "open_check"
            hs["events"].append((a.street, kind, g, b))
            hs["theirs"] = a.street_total
            hs["opp_acted_street"] = True
            hs["opp_raised"] = hs["opp_raised"] or a.action in _RAISES
    hs["n"] = len(obs.action_history)
    return hs


def _note_my_action(obs, action, memory):
    """Remember what the opponent will face after our action (needed because a final fold
    or check-behind never shows up in a later observation)."""
    hs = memory.get("hand")
    if hs is None:
        return
    if action in _RAISES:
        hs["pending"] = ("face", 2 * int(obs.street != Street.PREFLOP) + int(hs["opp_raised"]))
    elif action == Action.CHECK_CALL and obs.to_call == 0 and not hs["opp_acted_street"] \
            and obs.street != Street.PREFLOP:
        hs["pending"] = ("open", 1)
    else:
        hs["pending"] = None


# ---------------------------------------------------------------------------
# Particle filter over the opponent's hand + equity
# ---------------------------------------------------------------------------
def _budget(memory):
    """Compute budget for this half: BANK_TARGET_S minus the projected request overhead."""
    played = memory["hands_played"]
    per_hand = memory["calls"] / played if played >= 20 else 3.6
    overhead = per_hand * memory["num_hands"] * OVERHEAD_MS / 1000.0
    return min(BANK_BUDGET_S, max(1.5, BANK_TARGET_S - overhead))


def _pool_size(obs, memory):
    if memory["t_used"] >= _budget(memory):  # over budget: smallest pool from here on
        return POOL_MIN // 2
    return int(memory["pool_n"]) // 25 * 25


def _govern(memory, hand_time, used_pool):
    """Feedback control of the pool size: compare what hands really cost with what the
    remaining budget allows per remaining hand, and nudge the pool size up or down."""
    memory["np_frac"] = 0.97 * memory["np_frac"] + 0.03 * float(used_pool)
    if used_pool:
        memory["np_hand_s"] = 0.9 * memory["np_hand_s"] + 0.1 * hand_time
    remaining = max(1, memory["num_hands"] - memory["hands_played"])
    target = max(0.0, _budget(memory) - memory["t_used"]) / remaining        # seconds per hand
    expected = memory["np_frac"] * memory["np_hand_s"] + 1e-5
    step = min(1.15, max(0.85, (target / expected) ** 0.5))
    memory["pool_n"] = min(POOL_MAX, max(POOL_MIN, memory["pool_n"] * step))


def _init_pool(obs, memory, n):
    """Take n opponent hands from the precomputed bank that don't use our cards."""
    k = len(_OB_HANDS)
    off = int(memory["np_rng"].integers(0, k))
    rows = np.arange(off, off + 3 * n) % k
    rows = rows[~_OB_MASK[rows][:, list(obs.hole_cards)].any(1)][:n]
    n = len(rows)
    return {"n": n, "opp": _OB_HANDS[rows], "mask": _OB_MASK[rows], "pk": _OB_PK[rows],
            "pre": _OB_PRE[rows], "my_t": _my_table(obs.hole_cards), "w": np.ones(n),
            "valid": np.ones(n, bool), "nb": 0, "x": {}, "river": None, "applied": 0, "t": 0.0}


FD_PCT = (0.62, 0.55)     # strength we credit a flush draw with on the flop / turn


def _strength(pf, obs, street):
    """Opponent hand strength (percentile 0..1) as seen at `street`, cached per street:
    preflop = fitted formula; later = made hand on that board, flush draws credited."""
    if street in pf["x"]:
        return pf["x"][street]
    valid = pf["valid"]
    if street == Street.PREFLOP:
        x = pf["pre"]
    else:
        k = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
        board = list(obs.board[:k])
        if k == 5:
            if pf["river"] is None:
                pf["river"] = _best(pf["pk"], _trip_keys(np.array(board)))
            x = _pct(pf["river"], valid)
        else:
            x = _pct(_best(pf["pk"], _partial_trip_keys(board)), valid)
            suits = [c & 3 for c in board]
            draw_suits = [s for s in range(4) if suits.count(s) == 2]
            if draw_suits:
                sc = (pf["pk"] // _NT) % 5 - 1          # suit of each suited hole pair, -1 if none
                fd = sc == draw_suits[0]
                for d in draw_suits[1:]:
                    fd |= sc == d
                fd = fd.any(1)
                x = np.where(fd, np.maximum(x, FD_PCT[k - 3]), x)
    pf["x"][street] = x
    return x


def _likelihood(kind, x, memory, g, b):
    fold, call, raise_ = _face_freqs(memory, b)
    if kind == "open_bet":
        return BLUFF_FLOOR + (1 - BLUFF_FLOOR) * _sig((x - (1 - _bet_freq(memory, g))) / RANGE_TAU)
    if kind == "open_check":
        return 1.0 - 0.7 * _sig((x - (1 - _bet_freq(memory, g))) / RANGE_TAU)
    if kind == "face_raise":
        return BLUFF_FLOOR + (1 - BLUFF_FLOOR) * _sig((x - (1 - raise_)) / RANGE_TAU)
    if kind == "face_call":
        return _sig((x - fold) / RANGE_TAU) * (1.0 - 0.6 * _sig((x - (1 - raise_)) / RANGE_TAU))
    return np.ones_like(x)


def _update_pool(obs, memory, hs):
    if hs["pf"] is None:
        hs["pf"] = _init_pool(obs, memory, _pool_size(obs, memory))
    pf = hs["pf"]
    if len(obs.board) > pf["nb"]:           # new board cards: drop opponent hands that hold them
        hit = pf["mask"][:, list(obs.board[pf["nb"]:])].any(1)
        pf["valid"] &= ~hit
        pf["w"][hit] = 0.0
        pf["nb"] = len(obs.board)
        pf["x"] = {s: v for s, v in pf["x"].items() if s == Street.PREFLOP}
    for street, kind, g, b in hs["events"][pf["applied"]:]:
        x = _strength(pf, obs, street)
        pf["w"] = pf["w"] * _likelihood(kind, x, memory, g, b)
    pf["applied"] = len(hs["events"])
    return pf


def _results(obs, pf, rng):
    """Our result (1 win, 0.5 tie, 0 loss) against every pooled opponent hand, on a random
    run-out drawn for that hand. Cached for the street (only the weights change)."""
    key = len(obs.board)
    if pf.get("res_key") != key:
        pf["res"], pf["res_key"] = _results_now(obs, pf, rng), key
    return pf["res"]


def _results_now(obs, pf, rng):
    need = 5 - len(obs.board)
    if need == 0:
        tk = _trip_keys(np.array(obs.board))
        mine = pf["my_t"][tk].max()
        if pf["river"] is None:
            pf["river"] = _best(pf["pk"], tk)
        theirs = pf["river"]
    else:
        dead = set(obs.hole_cards) | set(obs.board)
        if need <= 2:
            run = _draw_few(pf["mask"], np.array([c for c in range(36) if c not in dead]), need, rng)
        else:
            alive = ~pf["mask"]
            alive[:, list(dead)] = False
            run = _draw(alive, need, rng)
        if obs.board:
            run = np.concatenate([np.broadcast_to(np.array(obs.board), (pf["n"], len(obs.board))), run], 1)
        tk = _trip_keys(run)
        mine = pf["my_t"][tk].max(1)
        theirs = _best(pf["pk"], tk)
    return (mine > theirs) + 0.5 * (mine == theirs)


def _norm_weights(pf):
    w = pf["w"] * pf["valid"]
    s = w.sum()
    if s <= 0:
        w = pf["valid"].astype(float)
        s = max(1.0, w.sum())
    w = w / s
    ess = 1.0 / max(1e-12, (w * w).sum())
    if ess < 40:                            # model too sure of itself: blend toward uniform
        u = pf["valid"] / max(1, pf["valid"].sum())
        w = 0.5 * w + 0.5 * u
    return w


def _split_equity(w, x, res, fold, reraise, order, cw):
    """Split the range by strength: the weakest `fold` share folds, the strongest `reraise`
    share re-raises, the rest calls. Returns (equity vs callers, equity vs re-raisers)."""
    xs = x[order]
    top = len(order) - 1
    q_lo = xs[min(top, int(np.searchsorted(cw, fold)))] if fold > 0.01 else -1.0
    q_hi = xs[min(top, int(np.searchsorted(cw, 1.0 - reraise)))] if reraise > 0.01 else 2.0
    hi = _sig((x - q_hi) / 0.04)
    wc = w * _sig((x - q_lo) / 0.04) * (1.0 - hi)
    wr = w * hi
    base = float((w * res).sum())
    sc, sr = wc.sum(), wr.sum()
    return (float((wc * res).sum() / sc) if sc > 1e-9 else base,
            float((wr * res).sum() / sr) if sr > 1e-9 else base)


def _decide_np(obs, memory, hs):
    if obs.street == Street.PREFLOP and not hs["opp_raised"]:
        # Unraised preflop: their range is still ~random, so the fitted formula IS our equity
        # vs it (10 us instead of a 5-card Monte Carlo), and we never open-raise preflop.
        if obs.to_call == 0:
            return Action.CHECK_CALL
        tax = CALL_MARGIN * (1.0 if obs.is_button else 2.0)
        call_ev = (preflop_equity(obs.hole_cards) - tax) * (obs.pot + obs.to_call) - obs.to_call
        return Action.CHECK_CALL if call_ev >= 0 else Action.FOLD
    pf = _update_pool(obs, memory, hs)
    rng = memory["np_rng"]
    res = _results(obs, pf, rng)
    w = _norm_weights(pf)
    eq = float((w * res).sum())
    g = 0 if obs.street == Street.PREFLOP else 1
    x = _strength(pf, obs, obs.street)
    fold, _, reraise = _face_freqs(memory, 2 * g + int(hs["opp_raised"]))

    pot, to_call = obs.pot, obs.to_call
    last = obs.street == Street.RIVER
    call_all_in = to_call >= obs.my_stack
    oop = not obs.is_button                 # the button acts last after the flop
    tax = 0.0 if (last or call_all_in) else CALL_MARGIN * (2.0 if oop else 1.0)

    # expected chips from here on, relative to folding now
    evs = {}
    if to_call > 0:
        evs[Action.FOLD] = 0.0
        evs[Action.CHECK_CALL] = (eq - tax) * (pot + to_call) - to_call
    else:
        evs[Action.CHECK_CALL] = eq * pot
    passive = max(evs.values())
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    edge = RAISE_EDGE_PRE if obs.street == Street.PREFLOP else RAISE_EDGE
    for act in _RAISES:
        if act not in obs.legal_actions:
            continue
        add = obs.raise_to[act] - obs.my_bet                 # chips we put in now
        opp_call = add - to_call                             # chips the opponent must add to call
        f = fold * (1 + SIZE_FOLD if act == Action.RAISE_POT else 1 - SIZE_FOLD) if obs.opp_stack > 0 else 0.0
        f = min(0.9, f)
        # can they raise back? (raise cap 4 per street, and they need chips beyond the call)
        can_rr = obs.raises_this_street + 1 < 4 and obs.opp_stack > opp_call and obs.my_stack > add
        r = min(reraise, 0.95 - f) if can_rr else 0.0
        eqc, eqr = _split_equity(w, x, res, f, r, order, cw)
        called_pot = pot + add + opp_call
        ev_called = eqc * called_pot - add
        if r > 0:
            rr = min(called_pot, obs.my_stack - add, obs.opp_stack - opp_call)   # their pot-size re-raise
            ev_rr = max(-add, eqr * (called_pot + 2 * rr) - add - rr)            # we fold or call it
        else:
            ev_rr = 0.0
        ev = f * pot + (1 - f - r) * ev_called + r * ev_rr
        if ev > passive + edge * pot:
            evs[act] = ev
    best = max(evs, key=evs.get)
    if best == Action.CHECK_CALL and to_call > 0 and evs[Action.FOLD] >= evs[Action.CHECK_CALL]:
        best = Action.FOLD
    return best


# ---------------------------------------------------------------------------
# v3 fallback (pure Python thresholds), used only if numpy is unavailable
# ---------------------------------------------------------------------------
def _opp_stats(memory):
    raise_rate = (memory["opp_raises"] + 3) / (memory["opp_actions"] + 10)
    fold_rate = (memory["opp_folds_to_me"] + 3) / (memory["my_raise_hands"] + 8)
    return raise_rate, fold_rate


def _update_from_history(obs, memory):
    if obs.hand_index != memory["seen_hand"]:
        memory["seen_hand"] = obs.hand_index
        memory["seen_actions"] = 0
        memory["i_raised_hand"] = False
    hist = obs.action_history
    for a in hist[memory["seen_actions"]:]:
        if a.is_me:
            if a.action in _RAISES and not memory["i_raised_hand"]:
                memory["i_raised_hand"] = True
                memory["my_raise_hands"] += 1
        else:
            memory["opp_actions"] += 1
            if a.action in _RAISES:
                memory["opp_raises"] += 1
    memory["seen_actions"] = len(hist)


def _sample_count(obs, memory) -> int:
    remaining_hands = max(1, memory["num_hands"] - obs.hand_index)
    left_s = BANK_BUDGET_S - memory["t_used"]
    if left_s <= 0:
        return MIN_SAMPLES
    per_decision_ms = left_s / (remaining_hands * 2.4) * 1000.0
    weight = 0.5 if obs.pot <= 8 else (1.0 if obs.pot <= 30 else 2.0)
    ms = min(MAX_DECISION_MS, per_decision_ms * weight)
    n = int(ms * 1000.0 / memory["us_per_sample"])
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


def _equity(obs, memory) -> float:
    if obs.street == Street.PREFLOP:
        return preflop_equity(obs.hole_cards)
    n = _sample_count(obs, memory)
    t0 = time.perf_counter()
    e = mc_equity(obs.hole_cards, obs.board, n, memory["rng"])
    memory["us_per_sample"] = 0.9 * memory["us_per_sample"] + 0.1 * ((time.perf_counter() - t0) / n * 1e6)
    return e


def _decide_simple(obs: Observation, memory: dict) -> Action:
    legal = obs.legal_actions
    can_half = Action.RAISE_HALF in legal
    can_pot = Action.RAISE_POT in legal
    pre = obs.street == Street.PREFLOP
    raise_rate, fold_rate = _opp_stats(memory)
    e = _equity(obs, memory)

    def raise_with(want_pot: bool) -> Action:
        if want_pot and can_pot:
            return Action.RAISE_POT
        return Action.RAISE_HALF if can_half else Action.RAISE_POT

    def commits_too_much(act: Action) -> bool:
        amount = obs.raise_to.get(act, 0) - obs.my_bet
        return obs.my_stack > 0 and amount / (obs.my_stack + 0.0) > 0.5 and e < 0.70

    if obs.to_call == 0:
        half_t, pot_t = OPEN_PRE if pre else OPEN_POST
        if (can_half or can_pot) and e >= half_t:
            act = raise_with(e >= pot_t)
            if not commits_too_much(act):
                return act
        if can_half and fold_rate > 0.55 and memory["rng"].random() < BLUFF_RATE:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    price = obs.to_call / (obs.pot + obs.to_call)
    margin = BASE_MARGIN + 0.025 * obs.raises_this_street + 0.05 * min(1.0, obs.to_call / max(1, obs.pot - obs.to_call))
    if raise_rate > 0.5:
        margin -= 0.03
    elif raise_rate < 0.15:
        margin += 0.03
    half_t, pot_t = RERAISE_PRE if pre else RERAISE_POST
    if (can_half or can_pot) and e >= half_t:
        act = raise_with(e >= pot_t)
        if not commits_too_much(act):
            return act
    if e >= price + margin:
        return Action.CHECK_CALL
    return Action.FOLD


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _panic(obs):
    """Zero-compute play for when the time bank is nearly used up (an empty bank means
    check/fold for the rest of the half, which is far worse)."""
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if obs.street == Street.PREFLOP:
        eq = preflop_equity(obs.hole_cards)
        return Action.CHECK_CALL if eq * (obs.pot + obs.to_call) >= obs.to_call else Action.FOLD
    return Action.CHECK_CALL if obs.to_call * 4 <= obs.pot else Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.
    Any crash falls back to a legal safe action so a bug never counts as an error."""
    t0 = time.perf_counter()
    try:
        _update_from_history(obs, memory)
        action = None
        memory["calls"] += 1
        charged = memory["t_used"] + memory["calls"] * OVERHEAD_MS / 1000.0
        if charged >= BANK_SAFE_S:          # time bank nearly gone: never let it run out
            action = _panic(obs)
        elif HAVE_NP:
            try:
                hs = _scan_history(obs, memory)
                action = _decide_np(obs, memory, hs)
            except Exception:
                traceback.print_exc()
                action = None
        if action is None:
            action = _decide_simple(obs, memory)
        if action not in obs.legal_actions:
            raise ValueError("illegal action chosen")
    except Exception:
        traceback.print_exc()
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    try:
        _note_my_action(obs, action, memory)
    except Exception:
        traceback.print_exc()
    dt = time.perf_counter() - t0
    memory["t_used"] += dt
    hs = memory.get("hand")
    if hs is not None and hs.get("pf") is not None:
        hs["pf"]["t"] += dt
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand: update the opponent model."""
    memory["hands_played"] += 1
    if memory["i_raised_hand"] and not result.went_to_showdown and result.my_net > 0:
        memory["opp_folds_to_me"] += 1
    hs = memory.get("hand")
    if hs is None or hs["idx"] != result.hand_index:
        return
    # the opponent's last action is never shown to us: infer it from how the hand ended
    pend = hs.get("pending")
    if pend is not None:
        kind, g = pend
        if kind == "face":                  # here g is the o_face bucket
            if not result.went_to_showdown and result.my_net > 0:
                memory["o_face"][g][0] += 1
            elif result.went_to_showdown:
                memory["o_face"][g][1] += 1
        elif kind == "open" and result.went_to_showdown:
            memory["o_open"][g][0] += 1
    pf = hs.get("pf")
    _govern(memory, pf["t"] if pf is not None else 0.0, pf is not None)
    hs["pending"] = None

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
