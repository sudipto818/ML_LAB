# ML_LAB

Sudipto Ghosh (240150042)
Badri Bishal Das (240150006)

DA322 Machine Learning Lab, IIT Guwahati.

Our project is LOT (Lookahead Oblique Trees). It is a decision tree that mixes two ideas:

- lookahead from SPLIT, so it can pick a split that only pays off one level later (like XOR)
- sparse oblique splits from SPORF, like x1 + x2 <= t, so it can cut diagonal boundaries

The code is in `LOT/`, the papers are in `Papers/`.
