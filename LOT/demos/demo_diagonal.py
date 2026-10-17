import numpy as np
from lot.data import diagonal
from lot.tree import LOT

"""Diagonal: y = 1[x1 + x2 > 0]. One oblique split is exact; an axis tree needs a staircase"""
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
