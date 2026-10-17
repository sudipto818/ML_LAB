import numpy as np
from lot.data import parity
from lot.tree import LOT, gains, oneHot
from demos.style import plt, save, scatterClasses, blue, orange, ink2

"""XOR: y = 1[x1 > 0] XOR 1[x2 > 0]"""
X, y = parity(1000, 2, 0, 1)
Xt, yt = parity(4000, 2, 0, 101)

for name, la in [("Greedy", False), ("Look", True)]:
    m = LOT(depth=2, lookahead=la).fit(X, y)
    print("==", name, "==")
    m.show()
    print("leaves:", m.nLeaves(), "| train objective R:", round(m.cost, 4),
          "| test error:", round(float(np.mean(m.predict(Xt) != yt)), 3))
    print()

"""Figure 1 of the spec: (a) the data, (b) information gain against threshold, computed by gains()
on 20000 simulated points: every root split x1 <= tau has gain ~0, but inside {x1 <= 0} the split
x2 <= tau reaches ~1 bit at tau = 0 (Lemma 6)"""
Xg, yg = parity(20000, 2, 0, 7)
Yg = oneHot(yg, 2)
taus = np.linspace(-0.98, 0.98, 99)
root = gains(np.stack([Xg[:, 0] <= t for t in taus], axis=1), Yg)
left = np.flatnonzero(Xg[:, 0] <= 0)
second = gains(np.stack([Xg[left, 1] <= t for t in taus], axis=1), Yg[left])

fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
scatterClasses(a, X[:450], y[:450], 0, 1)
a.axhline(0, color=ink2, lw=0.8)
a.axvline(0, color=ink2, lw=0.8)
a.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="$x_1$", ylabel="$x_2$", title="(a) XOR data, 450 points")
a.set_aspect("equal")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
b.plot(taus, second, color=orange, label=r"second split $x_2 \leq \tau$ inside $\{x_1 \leq 0\}$")
b.plot(taus, root, color=blue, label=r"root split $x_1 \leq \tau$")
k = int(np.argmax(second))
b.annotate(f"{second[k]:.2f} bits at $\\tau$ = {taus[k]:.2f}", (taus[k], second[k]), xytext=(0.25, 0.9),
           textcoords="data", fontsize=8, color=ink2, arrowprops={"arrowstyle": "-", "color": ink2, "lw": 0.6})
b.set(xlim=(-1, 1), ylim=(-0.02, 1.05), xlabel=r"threshold $\tau$", ylabel="information gain (bits)",
      title="(b) Information gain against threshold")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
fig.tight_layout()
save(fig, "xor.png")
