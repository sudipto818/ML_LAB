import numpy as np
from lot.data import parity
from lot.tree import LOT

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