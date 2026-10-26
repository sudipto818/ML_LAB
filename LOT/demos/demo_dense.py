import zlib
import numpy as np
import lot.tree
from lot.data import diagonal
from lot.oblique import sampleProjections
from lot.tree import LOT
from demos.style import plt, save, blue, orange, aqua, ink2

"""SPORF: with noise features, sparse directions (||a||_0 <= 3) beat dense random ones. Greedy oblique
trees (SPORF's selection rule) on diagonal with p_noise noise features, depth 4, 5 seeds"""


def denseProjections(p, s, m, idx, level, seed):
    """the same budget as sparse, m per support size 2..s, but every direction uses all p features"""
    rng = np.random.default_rng([zlib.crc32(idx.tobytes()), level, seed])
    out = []
    for k in range(m * (s - 1)):
        out.append((tuple(range(p)), (1,) + tuple(rng.choice((1, -1), p - 1).tolist())))
    return out


noises = (0, 2, 4, 8, 16, 32)
seeds = range(5)
err = {"axis": [], "sparse": [], "dense": []}
for noise in noises:
    e = {"axis": [], "sparse": [], "dense": []}
    for seed in seeds:
        X, y = diagonal(1000, noise, seed)
        Xt, yt = diagonal(20000, noise, 100 + seed)
        e["axis"].append(np.mean(LOT(depth=4).fit(X, y).predict(Xt) != yt))
        e["sparse"].append(np.mean(LOT(depth=4, oblique=True, seed=seed).fit(X, y).predict(Xt) != yt))
        lot.tree.sampleProjections = denseProjections
        e["dense"].append(np.mean(LOT(depth=4, oblique=True, seed=seed).fit(X, y).predict(Xt) != yt))
        lot.tree.sampleProjections = sampleProjections
    for k in e:
        err[k].append((np.mean(e[k]), np.std(e[k])))
    print("p_noise", noise, " ".join("%s %.3f +- %.3f" % (k, err[k][-1][0], err[k][-1][1]) for k in err))

fig, ax = plt.subplots(figsize=(5.2, 3.4))
x = np.array(noises)
for k, c, mk, ls in [("axis", blue, "o", "-"), ("sparse", orange, "s", "-"), ("dense", aqua, "D", (0, (1.5, 1.5)))]:
    mu = np.array([v[0] for v in err[k]])
    sd = np.array([v[1] for v in err[k]])
    lab = {"axis": "axis only", "sparse": r"sparse, $\|a\|_0 \leq 3$", "dense": "dense, all p features"}[k]
    ax.errorbar(x, mu, yerr=sd, color=c, marker=mk, ms=5, ls=ls, capsize=2, label=lab)
ax.set_xscale("symlog", linthresh=2)
ax.set_xticks(noises, [str(n) for n in noises])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.axvline(6, color=ink2, lw=0.8, ls=(0, (2, 2)))
ax.text(6.6, 0.028, "pairs sampled:\n$|P_2| = p(p-1) > 30$", fontsize=7.5, color=ink2, va="top", ha="left")
ax.set(xlabel="noise features added to diagonal", ylabel="test error (mean, sd over 5 seeds)",
       title="Sparse vs dense oblique splits (greedy, depth 4)", ylim=(-0.01, None))
ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "dense_vs_sparse.png")
