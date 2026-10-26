import math
import numpy as np
from lot.data import rotatedParity
from lot.oblique import nProjections
from lot.tree import LOT
from demos.style import plt, save, blue, orange, aqua, ink2

"""Remark 16: LOT finds the Figure 4 tree only if the root draws a correct pair and both children draw
the other one. Rotated parity with p = 10 (6 noise features), pairs only (s = 2), depth 2, 10 seeds per m"""
X, y = rotatedParity(1000, 6, 1)
Xt, yt = rotatedParity(20000, 6, 101)
P2 = nProjections(10, 2)
ms = (5, 10, 20, 30, 45, 60, 90)
seeds = range(10)


def found(tree):
    """root on one correct pair, both children on the other"""
    if tree[0] == "leaf" or tree[2][0] == "leaf" or tree[3][0] == "leaf":
        return False
    pairs = {((0, 1), (1, 1)), ((2, 3), (1, -1))}
    root = tree[1][:2]
    other = (pairs - {root}).pop() if root in pairs else None
    return other is not None and tree[2][1][:2] == other and tree[3][1][:2] == other


rate = []
errs = []
for m in ms:
    hit = []
    e = []
    for seed in seeds:
        lot = LOT(depth=2, lookahead=True, oblique=True, s=2, m=m, seed=seed).fit(X, y)
        hit.append(found(lot.tree))
        e.append(np.mean(lot.predict(Xt) != yt))
    rate.append(np.mean(hit))
    errs.append((np.mean(e), np.std(e)))
    print("m", m, "pi2 %.3f" % min(1, m / P2), "found", sum(hit), "/ 10", "test error %.3f +- %.3f" % errs[-1])

"""spec: each of the three draws succeeds with pi2, so pi2^3; but the root may use either correct pair,
so its chance is 1 - C(P2-2, m)/C(P2, m), and only the children need one specific pair"""
mm = np.arange(1, P2 + 1)
pi = np.minimum(1, mm / P2)
root = np.array([1 - math.comb(P2 - 2, int(k)) / math.comb(P2, int(k)) for k in mm])

fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.5))
a.plot(mm, pi ** 3, color=blue, ls=(0, (4, 2)), label=r"spec Remark 16: $\pi_2^3$")
a.plot(mm, root * pi ** 2, color=blue, label=r"either pair at the root: $(1-\binom{88}{m}/\binom{90}{m})\,\pi_2^2$")
a.plot(ms, rate, "o", color=orange, ms=7, label="observed, 10 seeds")
a.set(xlabel=r"projections drawn per node, $m$  ($|P_2| = 90$)", ylabel="fraction of fits finding Figure 4",
      title="(a) Coverage compounds", ylim=(-0.03, 1.05))
a.legend(loc="upper left")
b.errorbar(ms, [v[0] for v in errs], yerr=[v[1] for v in errs], color=aqua, marker="D", ms=5, capsize=2)
b.set(xlabel=r"projections drawn per node, $m$", ylabel="test error (mean, sd)", title="(b) LOT test error",
      ylim=(-0.01, None))
for ax in (a, b):
    top = ax.secondary_xaxis("top", functions=(lambda v: v / P2, lambda v: v * P2))
    top.set_xlabel(r"$\pi_2 = m/|P_2|$", color=ink2)
fig.tight_layout()
save(fig, "projections.png")
