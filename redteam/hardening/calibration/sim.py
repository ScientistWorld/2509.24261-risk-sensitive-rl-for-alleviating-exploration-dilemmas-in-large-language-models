"""Calibration simulator for the hidden-scale redesign of the bandit gym.

In-process mirror of the evaluator's episode loop (same sampling semantics,
per-group hidden reward offsets, advantage cap, centring and softmax
policy-gradient step), with a configurable scaled-instance generator. Used to
choose the instance distribution before the final job; it is evidence, not
part of the gym's scoring. Rules are plain advantage functions with state.

Usage: python sim.py CONFIG_NAME [episodes] [num_batches] > out.jsonl
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time

BATCH, LR, CAP, OFFSET_W = 16, 0.1, 8.0, 2.0
GP_THRESHOLD = 0.9


def bell(d: float, w: float = 5.0) -> float:
    return math.exp(-(d * d) / (2 * w * w))


def seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, ("calib",) + parts)).encode()).digest()[:16], "big")


# ----------------------------------------------------------------- policy
class Tree:
    """Softmax policy with a segment tree of exp-weights: O(log n) sampling
    and updates; parent sums are recomputed from their children (no drift)."""

    def __init__(self, logits):
        self.n = len(logits)
        size = 1
        while size < self.n:
            size *= 2
        self.size = size
        self.theta = list(logits)
        self.rebuild()

    def rebuild(self):
        self.shift = max(self.theta)
        size = self.size
        t = [0.0] * (2 * size)
        t[size:size + self.n] = [math.exp(x - self.shift) for x in self.theta]
        for i in range(size - 1, 0, -1):
            t[i] = t[2 * i] + t[2 * i + 1]
        self.t = t

    def sample(self, rng, k):
        t, size, out = self.t, self.size, []
        for _ in range(k):
            x = rng.random() * t[1]
            i = 1
            while i < size:
                left = t[2 * i]
                if (x < left and left > 0.0) or t[2 * i + 1] == 0.0:
                    i = 2 * i
                else:
                    x -= left
                    i = 2 * i + 1
            out.append(i - size)
        return out

    def apply(self, deltas):
        th = self.theta
        rebuild = False
        for a, d in deltas.items():
            th[a] += d
            if th[a] - self.shift > 50.0:
                rebuild = True
        if rebuild:
            self.rebuild()
            return
        t, size, sh = self.t, self.size, self.shift
        for a in deltas:
            j = size + a
            t[j] = math.exp(th[a] - sh)
            j >>= 1
            while j:
                t[j] = t[2 * j] + t[2 * j + 1]
                j >>= 1
        if t[1] < 1e-200:
            self.rebuild()

    def probs(self):
        top = max(self.theta)
        w = [math.exp(x - top) for x in self.theta]
        z = math.fsum(w)
        return [x / z for x in w]


# -------------------------------------------------------------- instances
def paper_instance(rng):
    n = 100
    r0 = [0.1 + 0.9 * bell(x - 40) + 0.5 * bell(x - 75) for x in range(n)]
    p0 = [math.exp(-0.5 * ((x - 75) / 5.0) ** 2) + 0.1 for x in range(n)]
    tot = sum(p0)
    l0 = [math.log(p / tot + 1e-8) for p in p0]
    mu = sum(l0) / n
    l0 = [v - mu for v in l0]
    order = list(range(n))
    rng.shuffle(order)
    pos = {a: i for i, a in enumerate(order)}
    rewards = [r0[order[i]] for i in range(n)]
    return dict(rewards=rewards, logits=[l0[order[i]] for i in range(n)], lo=0.0, span=1.0,
                opt=pos[40], loc=pos[75],
                gp=sorted(i for i, r in enumerate(rewards) if r >= GP_THRESHOLD),
                lp=sorted(pos[x] for x in range(70, 81)), info={})


def log_u(lo, hi, q):
    return math.exp(math.log(lo) + q * (math.log(hi) - math.log(lo)))


def scaled_instance(rng, q, cfg, n_arms):
    offs = list(range(-10, 11))
    qm, qs, qr = q
    m = log_u(*cfg["m"], qm)
    S = log_u(*cfg["S"], qs) if cfg["S"][0] < cfg["S"][1] else cfg["S"][0]
    rho = cfg["rho"][0] + qr * (cfg["rho"][1] - cfg["rho"][0])
    bg = rng.uniform(0.0, 1.0 - S) if cfg.get("bg_random", True) else 0.1
    ladder = cfg.get("ladder")
    m2 = rho2 = 0.0
    if ladder:
        m2 = log_u(*ladder["m2"], rng.random())
        rho2 = rho + rng.uniform(*ladder["gap"]) * (1.0 - rho)
    free = 1.0 - m - m2
    p_local = rng.uniform(0.45, 0.65) * free
    p_decoy = rng.uniform(0.15, 0.25) * free
    p_tail = 1.0 - p_local - m - m2 - p_decoy
    per_arm = m / 21
    n_decoy = max(1, min(int(round(p_decoy / per_arm)), n_arms // 4))
    n_special = 21 * (3 if ladder else 2) + n_decoy
    special = rng.sample(range(n_arms), n_special)
    la, ga = special[:21], special[21:42]
    ia = special[42:63] if ladder else []
    de = special[63:] if ladder else special[42:]
    u = [0.0] * n_arms
    for d, a in zip(offs, la):
        u[a] = rho * bell(d)
    for d, a in zip(offs, ga):
        u[a] = bell(d)
    for d, a in zip(offs, ia):
        u[a] = rho2 * bell(d)
    probs = [0.0] * n_arms
    bw = [bell(d) for d in offs]
    bt = sum(bw)
    for w, a in zip(bw, la):
        probs[a] = p_local * w / bt
    for grp, mass in ((ga, m), (de, p_decoy), (ia, m2)):
        if not grp:
            continue
        ws = [math.exp(0.3 * rng.gauss(0, 1)) for _ in grp]
        t = sum(ws)
        for w, a in zip(ws, grp):
            probs[a] = mass * w / t
    taken = set(special)
    tw = [0.0] * n_arms
    for a in range(n_arms):
        if a not in taken:
            tw[a] = math.exp(rng.gauss(0, 1))
    tt = sum(tw)
    for a in range(n_arms):
        if a not in taken:
            probs[a] = p_tail * tw[a] / tt
    logits = [math.log(p) for p in probs]
    mu = sum(logits) / n_arms
    logits = [v - mu for v in logits]
    rewards = [bg + S * x for x in u]
    return dict(rewards=rewards, logits=logits, lo=bg, span=S, opt=ga[10], loc=la[10],
                gp=sorted(a for d, a in zip(offs, ga) if bell(d) >= GP_THRESHOLD),
                lp=sorted(a for d, a in zip(offs, la) if abs(d) <= 5),
                info=dict(m=m, S=S, rho=rho, m2=m2, rho2=rho2))


def strata(name, count):
    rng = random.Random(seed(name, "strata"))
    perms = []
    for _ in range(3):
        p = list(range(count))
        rng.shuffle(p)
        perms.append(p)
    return [tuple((perms[k][i] + rng.random()) / count for k in range(3)) for i in range(count)]


# ------------------------------------------------------------------ rules
def rs_adv(r, beta):
    n = len(r)
    if beta == 0:
        mu = sum(r) / n
        return [x - mu for x in r]
    s = [beta * x for x in r]
    top = max(s)
    v = (math.log(sum(math.exp(x - top) for x in s) / n) + top) / beta
    return [(math.exp(beta * (x - v)) - 1) / beta for x in r]


class Rule:
    def start(self, n_arms):
        self.n_arms = n_arms


class RS(Rule):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, r):
        return rs_adv(r, self.beta)


class ScaledRS(Rule):
    """Eq. 8 multiplied by a constant (a pure step-size change)."""

    def __init__(self, beta, scale):
        self.beta, self.scale = beta, scale

    def __call__(self, r):
        return [max(-CAP, min(CAP, self.scale * a)) for a in rs_adv(r, self.beta)]


class Anchored(Rule):
    """Success-only ported to offsets: +8 if r - group min >= tau, else -1.
    tau_paper is used on the 100-arm experiment (public scale 0.9 there)."""

    def __init__(self, tau, tau_paper=0.75):
        self.tau, self.tau_paper = tau, tau_paper

    def __call__(self, r):
        lo = min(r)
        tau = self.tau_paper if self.n_arms == 100 else self.tau
        return [CAP if x - lo >= tau else -1.0 for x in r]


class MaxSeek(Rule):
    """Scale-free: +8 for samples within tol of the best anchored reward seen."""

    def __init__(self, tol=0.05, wait=0):
        self.tol, self.wait = tol, wait

    def start(self, n_arms):
        super().start(n_arms)
        self.M, self.t = 0.0, 0

    def __call__(self, r):
        lo = min(r)
        x = [v - lo for v in r]
        self.M = max(self.M, max(x))
        self.t += 1
        if self.t <= self.wait or self.M <= 0:
            return [0.0] * len(r)
        return [CAP if v >= self.M * (1 - self.tol) else -1.0 for v in x]


class AdaptiveRS(Rule):
    """Eq. 8 on anchored rewards divided by the running max spread."""

    def __init__(self, beta, scale=1.0):
        self.beta, self.scale = beta, scale

    def start(self, n_arms):
        super().start(n_arms)
        self.M = 0.0

    def __call__(self, r):
        lo = min(r)
        self.M = max(self.M, max(r) - lo)
        if self.M <= 0:
            return [0.0] * len(r)
        a = rs_adv([(v - lo) / self.M for v in r], self.beta)
        return [max(-CAP, min(CAP, self.scale * x)) for x in a]


class MemAspiration(Rule):
    """Exponential utility measured against the best anchored reward seen."""

    def __init__(self, beta=8.0, scale=16.0):
        self.beta, self.scale = beta, scale

    def start(self, n_arms):
        super().start(n_arms)
        self.M = 0.0

    def __call__(self, r):
        lo = min(r)
        x = [v - lo for v in r]
        self.M = max(self.M, max(x))
        if self.M <= 0:
            return [0.0] * len(r)
        e = [math.exp(self.beta * (v - self.M) / self.M) for v in x]
        mu = sum(e) / len(e)
        return [max(-CAP, min(CAP, self.scale * (v - mu) / self.beta)) for v in e]


class HistBest(Rule):
    """Reinforce only samples clearly above the usual best of a group
    (EMA of the anchored group maximum); scale-free novelty rule."""

    def __init__(self, margin=0.15, ema=0.01):
        self.margin, self.ema = margin, ema

    def start(self, n_arms):
        super().start(n_arms)
        self.L = None

    def __call__(self, r):
        lo = min(r)
        x = [v - lo for v in r]
        top = max(x)
        if self.L is None:
            self.L = top
        out = [CAP if v > (1 + self.margin) * self.L and v > 0 else -1.0 for v in x]
        self.L += self.ema * (top - self.L)
        return out


RULES = {
    "grpo": lambda: RS(0.0),
    "rs2": lambda: RS(2.0),
    "rs4": lambda: RS(4.0),
    "rs8": lambda: RS(8.0),
    "rs16": lambda: RS(16.0),
    "rs8x4": lambda: ScaledRS(8.0, 4.0),
    **{f"anch{t:g}": (lambda t=t: Anchored(t)) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75)},
    "maxseek": lambda: MaxSeek(0.05, 0),
    "etc300": lambda: MaxSeek(0.05, 300),
    "adaptrs8": lambda: AdaptiveRS(8.0),
    "adaptrs8x2": lambda: AdaptiveRS(8.0, 2.0),
    "memasp": lambda: MemAspiration(8.0, 16.0),
    "memasp_b16": lambda: MemAspiration(16.0, 32.0),
    "histbest": lambda: HistBest(0.15, 0.01),
}


# ---------------------------------------------------------------- episode
def run(inst, rule, ep_seed, nb):
    rng = random.Random(ep_seed)
    orng = random.Random(ep_seed * 7 + 1)
    pol = Tree(inst["logits"])
    R = inst["rewards"]
    lo, span = inst["lo"], inst["span"]
    units = [(x - lo) / span for x in R]
    rule.start(len(R))
    online = 0.0
    recent = []
    for _ in range(nb):
        acts = pol.sample(rng, BATCH)
        c = orng.uniform(-OFFSET_W, OFFSET_W)
        sc = [units[a] for a in acts]
        online += math.fsum(sc)
        recent.extend(sc)
        if len(recent) > 4000:
            recent = recent[-1000:]
        adv = rule([R[a] + c for a in acts])
        if any((not math.isfinite(v)) or abs(v) > CAP for v in adv):
            return None
        mu = math.fsum(adv) / BATCH
        d = {}
        for a, v in zip(acts, adv):
            d[a] = d.get(a, 0.0) + LR * (v - mu) / BATCH
        pol.apply(d)
    p = pol.probs()
    gp = math.fsum(p[a] for a in inst["gp"])
    tail = recent[-1000:]
    return dict(
        expected_reward=math.fsum(pi * ui for pi, ui in zip(p, units)),
        mean_reward_last_1000=math.fsum(tail) / len(tail),
        online_mean_reward=online / (nb * BATCH),
        optimal_action_probability=p[inst["opt"]],
        global_peak_probability=gp,
        global_peak_success_rate=1.0 if gp >= 0.5 else 0.0,
        local_action_probability=p[inst["loc"]],
        local_peak_probability=math.fsum(p[a] for a in inst["lp"]),
    )


CONFIGS = {
    # predecessor's draft: hidden span and local ratio
    "C1": dict(S=(0.3, 1.0), rho=(0.3, 0.85), m=(3e-4, 0.1)),
    "C2": dict(S=(0.5, 1.0), rho=(0.3, 0.85), m=(1e-3, 0.15)),
    "C3": dict(S=(0.3, 1.0), rho=(0.25, 0.75), m=(1e-3, 0.1),
               ladder=dict(m2=(0.01, 0.08), gap=(0.3, 0.7))),
    "C4": dict(S=(0.5, 1.0), rho=(0.25, 0.75), m=(1e-3, 0.15),
               ladder=dict(m2=(0.01, 0.08), gap=(0.3, 0.7))),
    "C5": dict(S=(0.3, 1.0), rho=(0.3, 0.85), m=(3e-3, 0.3)),
    "C6": dict(S=(0.3, 1.0), rho=(0.25, 0.75), m=(3e-3, 0.2),
               ladder=dict(m2=(0.02, 0.15), gap=(0.3, 0.7))),
    # old gym's scaled experiment (fixed span 0.9, local 5/9), with offsets
    "C0": dict(S=(0.9, 0.9), rho=(5 / 9, 5 / 9), m=(1e-3, 0.15), bg_random=False),
}


def main():
    name = sys.argv[1]
    eps = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 7500
    rules = sys.argv[4].split(",") if len(sys.argv) > 4 else list(RULES)
    n_arms = 2 ** 18
    t0 = time.time()
    if name == "E1":
        for ep in range(eps):
            inst = paper_instance(random.Random(seed("E1", ep)))
            for k in rules:
                res = run(inst, RULES[k](), seed("E1", ep, "s") % 2 ** 62, nb)
                print(json.dumps(dict(cfg="E1", ep=ep, rule=k, info={}, m=res)), flush=True)
            print(f"# E1 ep {ep} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        return
    cfg = CONFIGS[name]
    qs = strata(name, eps)
    for ep in range(eps):
        inst = scaled_instance(random.Random(seed(name, ep)), qs[ep], cfg, n_arms)
        for k in rules:
            res = run(inst, RULES[k](), seed(name, ep, "s") % 2 ** 62, nb)
            print(json.dumps(dict(cfg=name, ep=ep, rule=k, info=inst["info"], m=res)), flush=True)
        print(f"# {name} ep {ep} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
