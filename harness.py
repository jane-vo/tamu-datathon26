"""In-process duplicate matches using the real game.py (no web server needed)."""
import sys, types, random, math, time, importlib.util

def _stub_web():
    class _App:
        def get(self, *a, **k): return lambda f: f
        post = get
    fa = types.ModuleType("fastapi"); fa.FastAPI = lambda: _App(); fa.Request = object
    fr = types.ModuleType("fastapi.responses"); fr.JSONResponse = dict
    un = types.ModuleType("uvicorn"); un.run = lambda *a, **k: None
    sys.modules.update({"fastapi": fa, "fastapi.responses": fr, "uvicorn": un})

def load_agent(path="agent.py", name="agent_mod"):
    _stub_web()
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

from game import Game, Action, AgentConfig, make_deal

class ModBot:
    def __init__(self, mod): self.m = mod; self.mem = {}; self.t = 0.0
    def start(self): self.mem = {}; self.m.new_half(AgentConfig(), self.mem); self.t = 0.0
    def act(self, obs):
        t0 = time.perf_counter(); a = self.m.choose_action(obs, self.mem); self.t += time.perf_counter() - t0; return a
    def end(self, r): self.m.on_hand_end(r, self.mem)

class FuncBot:
    def __init__(self, fn): self.fn = fn; self.rng = random.Random(5); self.t = 0.0
    def start(self): self.rng = random.Random(5)
    def act(self, obs): return self.fn(obs, self.rng)
    def end(self, r): pass

def raises(obs): return [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
def random_opp(obs, rng):
    r = raises(obs)
    if r and rng.random() < 0.3: return rng.choice(r)
    return Action.CHECK_CALL if (obs.to_call == 0 or rng.random() < 0.8) else Action.FOLD
def station(obs, rng): return Action.CHECK_CALL
def maniac(obs, rng): return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
def checkfold(obs, rng): return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
def nit_raiser(obs, rng):  # raises ~15% of the time, folds a lot
    r = raises(obs)
    if r and rng.random() < 0.15: return Action.RAISE_HALF if Action.RAISE_HALF in r else r[0]
    if obs.to_call > 0 and rng.random() < 0.5: return Action.FOLD
    return Action.CHECK_CALL

def play_half(a, b, deals, seed, a_seat):
    """Seat a_seat is bot a. Returns list of a's chip net per deal."""
    a.start(); b.start(); nets = []
    bots = {a_seat: a, 1 - a_seat: b}
    for i in range(deals):
        g = Game(make_deal(seed, i), button_seat=i % 2, hand_index=i)
        while not g.is_terminal:
            s = g.to_act; obs = g.observation(s)
            act = bots[s].act(obs)
            assert act in obs.legal_actions, (act, obs.legal_actions)
            g.apply(act)
        for s in (0, 1): bots[s].end(g.hand_result(s))
        nets.append(g.result().nets[a_seat])
    return nets

def match(a, b, deals=300, seed="s1"):
    h1 = play_half(a, b, deals, seed, 0); t1 = a.t
    h2 = play_half(a, b, deals, seed, 1); t2 = a.t
    per = [(x + y) / 2.0 for x, y in zip(h1, h2)]  # big blinds per deal (BB = 2 chips)
    n = len(per); mu = sum(per) / n
    sd = math.sqrt(sum((p - mu) ** 2 for p in per) / (n - 1)); se = sd / math.sqrt(n)
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    verdict = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "CHOP")
    return mu, lo, hi, verdict, max(t1, t2)

if __name__ == "__main__":
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mod = load_agent("agent.py")
    opps = {"random": random_opp, "station": station, "maniac": maniac, "checkfold": checkfold, "nit": nit_raiser}
    for name, fn in opps.items():
        mu, lo, hi, v, t = match(ModBot(mod), FuncBot(fn), deals, seed=f"x-{name}")
        print(f"vs {name:9s} {mu:+.3f} bb/deal  CI [{lo:+.3f},{hi:+.3f}]  {v:5s}  my compute/half {t:.2f}s", flush=True)
