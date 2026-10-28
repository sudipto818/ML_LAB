# Timing

Exact wall-clock fit times, measured once with `python timing.py`.

- Machine: MacBookAir10,1, Apple M1, 4 performance + 4 efficiency cores, 8 GB RAM
- OS: macOS 15.7.7; Python 3.14.6, NumPy 2.5.4
- Power: Battery Power; one fit at a time, nothing else running
- depth 4, lambda = 0.005, gamma = lambda/2, the settings of the headline table

## Fit time per dataset (s, median of the 5 training folds of seed 0)

| Dataset | N train | p | Greedy axis | Look axis | Greedy oblique | LOT | LOT / Look axis |
|---|---|---|---|---|---|---|---|
| parity2 | 800 | 4 | 0.002 | 0.046 | 0.008 | 1.443 | 31x |
| parity3 | 800 | 5 | 0.003 | 0.086 | 0.028 | 10.325 | 120x |
| rotatedParity | 800 | 6 | 0.003 | 0.118 | 0.025 | 9.690 | 82x |
| diagonal | 800 | 4 | 0.002 | 0.060 | 0.006 | 0.723 | 12x |
| orthant | 800 | 5 | 0.002 | 0.090 | 0.028 | 10.549 | 118x |
| trunk | 800 | 10 | 0.003 | 0.219 | 0.032 | 15.284 | 70x |
| ringnorm | 1600 | 20 | 0.007 | 0.776 | 0.050 | 21.859 | 28x |
| balanceScale | 500 | 4 | 0.002 | 0.046 | 0.008 | 1.273 | 28x |
| vehicle | 676 | 18 | 0.004 | 0.461 | 0.036 | 17.487 | 38x |
| wdbc | 455 | 30 | 0.004 | 0.614 | 0.024 | 12.989 | 21x |
| compas | 1600 | 11 | 0.004 | 0.072 | 0.039 | 9.209 | 128x |

## Fit time as N grows (s, rotated parity with 2 noise features, p = 6)

| N | Greedy axis | Look axis | Greedy oblique | LOT |
|---|---|---|---|---|
| 250 | 0.002 | 0.077 | 0.015 | 7.264 |
| 500 | 0.002 | 0.106 | 0.023 | 8.301 |
| 1000 | 0.003 | 0.123 | 0.028 | 10.645 |
| 2000 | 0.004 | 0.163 | 0.047 | 16.634 |
| 4000 | 0.006 | 0.237 | 0.079 | 24.151 |
| exponent b in t ~ N^b | 0.46 | 0.39 | 0.58 | 0.45 |
