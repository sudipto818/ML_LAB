import numpy as np
from lot.data import rotatedParity
from lot.tree import LOT

"""Rotated parity: y = 1[x1 + x2 > 0] XOR 1[x3 - x4 > 0]. The right root split has zero gain,
so only lookahead with oblique candidates (LOT) finds the four-leaf tree at depth 2"""
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
