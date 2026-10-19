import numpy as np
from lot.data import diagonal
from lot.tree import LOT
from demos.style import plt, save, blue, orange, aqua, ink2, blueTint, orangeTint, wrong
from matplotlib.colors import ListedColormap

"""y = 1[x1 + x2 > 0]: one oblique split, or an axis staircase"""
X, y = diagonal(1000, 0, 1)
Xt, yt = diagonal(20000, 0, 101)

for name, la, ob in [("Greedy, axis", False, False), ("Look, axis", True, False),
                     ("Greedy, oblique", False, True), ("LOT", True, True)]:
    m = LOT(depth=4, lookahead=la, oblique=ob).fit(X, y)
    print("==", name, "==")
    m.show()
    print("leaves:", m.nLeaves(), "| train objective R:", round(m.cost, 4),
          "| test error:", round(float(np.mean(m.predict(Xt) != yt)), 3))
    print()

"""spec Figure 2; qAx = 5 alone stops the staircase at ~10 leaves (Remark 5), hence the qAx sweep"""
g = np.linspace(-1, 1, 401)
G1, G2 = np.meshgrid(g, g)
grid = np.stack([G1.ravel(), G2.ravel()], axis=1)
truth = (grid[:, 0] + grid[:, 1] > 0).astype(int)


def regions(ax, m, title):
    """tint = predicted class, grey = wrong, dashed = true boundary"""
    pred = m.predict(grid)
    img = np.where(pred != truth, 2, pred).reshape(G1.shape)
    ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap=ListedColormap([blueTint, orangeTint, wrong]),
              vmin=0, vmax=2, interpolation="nearest")
    ax.plot([-1, 1], [1, -1], ls="--", color=ink2, lw=1)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title)
    ax.grid(False)
    ax.set_aspect("equal")


axisTree = LOT(depth=4, lookahead=True).fit(X, y)
lot = LOT(depth=4, lookahead=True, oblique=True).fit(X, y)
pts = []
for qAx in (5, 15, 31):
    for lam in (0.0005, 0.002, 0.01, 0.05):
        m = LOT(depth=6, lam=lam, qAx=qAx, lookahead=True).fit(X, y)
        pts.append((m.nLeaves(), np.mean(m.predict(Xt) != yt)))
for S, e in pts:
    print("axis Look, depth 6: leaves", S, "test error", round(float(e), 4), "bound 1/(2S)", round(1 / (2 * S), 4))
eLot = np.mean(lot.predict(Xt) != yt)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
regions(a, axisTree, f"(a) Axis Look: {axisTree.nLeaves()} leaves")
regions(b, lot, f"(b) LOT: {lot.nLeaves()} leaves")
a.text(0, -1.42, f"test error = {np.mean(axisTree.predict(Xt) != yt):.3f}", ha="center", fontsize=8, color=ink2)
b.text(0, -1.42, f"test error = {eLot:.3f}", ha="center", fontsize=8, color=ink2)
S = np.arange(2, 65)
c.plot(S, 1 / (2 * S), color=blue, label="lower bound 1/(2S), Prop. 9")
c.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=orange, ms=6, label=r"axis Look trees, $\lambda$ and $q_{ax}$ sweep")
c.plot([lot.nLeaves()], [eLot], "D", color=aqua, ms=7, label="LOT, one oblique split")
c.set_xscale("log", base=2)
c.set_xticks([2, 4, 8, 16, 32, 64], ["2", "4", "8", "16", "32", "64"])
c.set(xlabel="number of leaves S", ylabel="test error", title="(c) Error against size", ylim=(-0.01, 0.27))
c.legend(loc="upper right")
fig.tight_layout()
save(fig, "diagonal.png")
