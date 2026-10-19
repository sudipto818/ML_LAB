import numpy as np
from lot.data import rotatedParity
from lot.tree import LOT, gains, oneHot
from demos.style import plt, save, scatterClasses, blue, orange, aqua

"""the right root split has zero gain, so only LOT finds the 4-leaf tree at depth 2"""
X, y = rotatedParity(1000, 0, 1)
Xt, yt = rotatedParity(20000, 0, 101)

for name, la, ob in [("Greedy, axis", False, False), ("Look, axis", True, False),
                     ("Greedy, oblique", False, True), ("LOT", True, True)]:
    m = LOT(depth=2, lookahead=la, oblique=ob).fit(X, y)
    print("==", name, "==")
    m.show()
    print("leaves:", m.nLeaves(), "| train objective R:", round(m.cost, 4),
          "| test error:", round(float(np.mean(m.predict(Xt) != yt)), 3))
    print()

"""spec Figure 3; gains on 40000 points (Lemma 15)"""
Xg, yg = rotatedParity(40000, 0, 7)
Yg = oneHot(yg, 2)
qs = np.linspace(0.01, 0.99, 99)


def rootGain(z):
    return gains(np.stack([z <= t for t in np.quantile(z, qs)], axis=1), Yg)


fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
scatterClasses(a, X[:450], y[:450], 0, 2)
a.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="$x_1$", ylabel="$x_3$", title="(a) Raw")
a.set_aspect("equal")
P = np.stack([X[:450, 0] + X[:450, 1], X[:450, 2] - X[:450, 3]], axis=1)
scatterClasses(b, P, y[:450], 0, 1)
b.axhline(0, color="#52514e", lw=0.8)
b.axvline(0, color="#52514e", lw=0.8)
b.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="$x_1 + x_2$", ylabel="$x_3 - x_4$", title="(b) Projected")
b.set_aspect("equal")
a.legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=2)
c.plot(qs, rootGain(Xg[:, 0] + Xg[:, 2]), color=orange, label=r"mixed: $x_1 + x_3 \leq \tau$")
c.plot(qs, rootGain(Xg[:, 0] + Xg[:, 1]), color=blue, label=r"correct: $x_1 + x_2 \leq \tau$")
c.plot(qs, rootGain(Xg[:, 0]), color=aqua, ls=(0, (1.5, 1.5)), label=r"axis: $x_1 \leq \tau$")
c.set(xlim=(0, 1), ylim=(-0.001, 0.032), xlabel=r"threshold $\tau$ as a quantile",
      ylabel="information gain (bits)", title="(c) Gain at the root")
c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
fig.tight_layout()
save(fig, "rotated_parity.png")
