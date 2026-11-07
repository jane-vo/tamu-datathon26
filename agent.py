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

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
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
PARTICIPANT = "YOUR_TEAM_NAME"
AGENT_NAME = "EquityBot-v1"


# ===== YOUR CODE HERE =====

# ---------------------------------------------------------------------------
# Tunable settings (change these while testing)
# ---------------------------------------------------------------------------
BANK_BUDGET_S = 3.5       # target for OUR OWN compute per half (the bank is 8 s and also pays for
                          # request overhead, so stay well below it; check the judge's "Faults" report)
MAX_DECISION_MS = 25.0    # never spend more than this on one decision (hard limit is 1000 ms)
MIN_SAMPLES = 10
MAX_SAMPLES = 400

# Raise thresholds on equity (our win chance vs a random hand): (half-pot, pot)
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
    """Monte Carlo win chance vs a random opponent hand: deal the missing board cards and an
    opponent hand n times and score both with the Omaha rule (ties count half)."""
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


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands). Set up everything we track."""
    memory["hands_played"] = 0
    memory["rng"] = random.Random(2026)
    memory["t_used"] = 0.0            # our own compute time this half (seconds)
    memory["us_per_sample"] = 60.0    # running estimate of one Monte Carlo sample (microseconds)
    memory["num_hands"] = config.num_hands
    # opponent model (all within this half)
    memory["opp_actions"] = 0         # opponent actions seen
    memory["opp_raises"] = 0          # ...of which raises
    memory["i_raised_hand"] = False   # did we raise in the current hand?
    memory["my_raise_hands"] = 0      # hands where we raised
    memory["opp_folds_to_me"] = 0     # of those, hands that ended with the opponent folding
    memory["seen_hand"] = -1          # hand_index of the hand whose history we last counted
    memory["seen_actions"] = 0        # how many history entries of that hand we counted


def _opp_stats(memory):
    """(opponent raise rate, opponent fold rate when we raise), with sensible priors."""
    raise_rate = (memory["opp_raises"] + 3) / (memory["opp_actions"] + 10)       # prior ~30%
    fold_rate = (memory["opp_folds_to_me"] + 3) / (memory["my_raise_hands"] + 8)  # prior ~38%
    return raise_rate, fold_rate


def _update_from_history(obs, memory):
    if obs.hand_index != memory["seen_hand"]:
        memory["seen_hand"] = obs.hand_index
        memory["seen_actions"] = 0
        memory["i_raised_hand"] = False
    hist = obs.action_history
    for a in hist[memory["seen_actions"]:]:
        if a.is_me:
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT) and not memory["i_raised_hand"]:
                memory["i_raised_hand"] = True
                memory["my_raise_hands"] += 1
        else:
            memory["opp_actions"] += 1
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raises"] += 1
    memory["seen_actions"] = len(hist)


def _sample_count(obs, memory) -> int:
    """How many simulations we can afford right now (spreads our budget over the half)."""
    remaining_hands = max(1, memory["num_hands"] - obs.hand_index)
    left_s = BANK_BUDGET_S - memory["t_used"]
    if left_s <= 0:
        return MIN_SAMPLES
    per_decision_ms = left_s / (remaining_hands * 2.4) * 1000.0
    pot = obs.pot
    weight = 0.5 if pot <= 8 else (1.0 if pot <= 30 else 2.0)  # big pots deserve more thinking
    ms = min(MAX_DECISION_MS, per_decision_ms * weight)
    n = int(ms * 1000.0 / memory["us_per_sample"])
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


def _equity(obs, memory) -> float:
    if obs.street == Street.PREFLOP:
        return preflop_equity(obs.hole_cards)
    n = _sample_count(obs, memory)
    t0 = time.perf_counter()
    e = mc_equity(obs.hole_cards, obs.board, n, memory["rng"])
    dt = time.perf_counter() - t0
    memory["us_per_sample"] = 0.9 * memory["us_per_sample"] + 0.1 * (dt / n * 1e6)
    return e


def _decide(obs: Observation, memory: dict) -> Action:
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
        """A raise that puts most of our stack in needs a strong hand."""
        amount = obs.raise_to.get(act, 0) - obs.my_bet
        return obs.my_stack > 0 and amount / (obs.my_stack + 0.0) > 0.5 and e < 0.70

    # ----- nothing to call: check or bet -----
    if obs.to_call == 0:
        half_t, pot_t = OPEN_PRE if pre else OPEN_POST
        if (can_half or can_pot) and e >= half_t:
            act = raise_with(e >= pot_t)
            if not commits_too_much(act):
                return act
        if can_half and fold_rate > 0.55 and memory["rng"].random() < BLUFF_RATE:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # ----- facing a bet: fold, call or raise -----
    price = obs.to_call / (obs.pot + obs.to_call)
    margin = BASE_MARGIN + 0.025 * obs.raises_this_street + 0.05 * min(1.0, obs.to_call / max(1, obs.pot - obs.to_call))
    if raise_rate > 0.5:       # very aggressive opponent: their bets mean less
        margin -= 0.03
    elif raise_rate < 0.15:    # passive opponent: respect their bets
        margin += 0.03
    half_t, pot_t = RERAISE_PRE if pre else RERAISE_POST
    if (can_half or can_pot) and e >= half_t:
        act = raise_with(e >= pot_t)
        if not commits_too_much(act):
            return act
    if e >= price + margin:
        return Action.CHECK_CALL
    return Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    Plan: estimate our win chance vs a random hand (preflop: fitted formula; later streets:
    Monte Carlo), compare it with the pot odds, and bet/raise/call/fold. Any crash falls back
    to a legal safe action so a bug never counts as an error.
    """
    t0 = time.perf_counter()
    try:
        _update_from_history(obs, memory)
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            raise ValueError("illegal action chosen")
    except Exception:
        traceback.print_exc()
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    memory["t_used"] += time.perf_counter() - t0
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand: update the opponent model."""
    memory["hands_played"] += 1
    # No showdown and we won chips = the opponent folded. Count it if we had raised.
    if memory["i_raised_hand"] and not result.went_to_showdown and result.my_net > 0:
        memory["opp_folds_to_me"] += 1

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
