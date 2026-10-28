# LOT: Lookahead Oblique Trees

A decision tree with SPLIT's lookahead and SPORF's sparse oblique splits.
Two switches give four variants: greedy or lookahead, axis or oblique. Lookahead + oblique is LOT.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From inside `LOT/`:

```bash
python -m pytest tests.py            # tests
python fetch.py                      # download the real datasets (needs curl)
python experiment.py                 # main experiment -> results/runs.csv
python ablations.py                  # ablations -> results/ablations.csv
python timing.py                     # fit times on this machine -> results/timing.md, 20 min
python plots.py                      # tables and figures
python demo.py                       # LOT vs the other 3 on rotated parity, 2 s
```

## Folders

- `lot/` the tree, data, projections, metrics
- `tests.py` the important checks
- `results/` csv files and figures

## Results

All numbers are measured, from `results/`. Settings: depth 4, lambda = 0.005, gamma = lambda/2,
5-fold stratified CV repeated with 3 seeds (15 runs per cell).

### Accuracy

Test accuracy (%), mean of 15 runs:

| Dataset | Greedy axis | Look axis | Greedy oblique | LOT |
|---|---|---|---|---|
| Rotated parity (synthetic) | 68.8 | 75.1 | 82.2 | **98.1** |
| Balance-scale | 74.2 | 75.8 | 84.8 | **87.0** |
| Vehicle | 65.7 | 67.3 | 67.8 | **70.7** |
| WDBC | **94.6** | **94.6** | 94.3 | 93.9 |
| COMPAS | 66.4 | 66.3 | **66.8** | 66.5 |

- Rotated parity needs both ideas: its first split is oblique and has zero gain. Only LOT solves it.
- On real data most of the gain comes from oblique splits. WDBC and COMPAS show no gain.
- Lookahead never did worse than greedy on the training objective: 1650 out of 1650 runs.

![test error of the four variants](results/figures/headline.png)

Blue is axis splits, orange is oblique. Hollow is greedy, filled is lookahead.
Panel (b) is the interaction I: below 0 means the two ideas help each other.

### Fit time

Measured with `python timing.py` on a MacBook Air (M1, 4 performance + 4 efficiency cores,
8 GB RAM, macOS 15.7, Python 3.14, NumPy 2.5), on battery, one fit at a time.
Full tables are in `results/timing.md`.

| | Greedy axis | Look axis | Greedy oblique | LOT |
|---|---|---|---|---|
| fit time per tree | 2 to 7 ms | 0.05 to 0.8 s | 6 to 50 ms | 0.7 to 22 s |

- LOT is the slowest variant, 12 to 128 times slower than Look axis.
  Look calls greedy twice for every candidate split, and oblique splits add a few hundred candidates per node.
- Time grows slowly with N (about N^0.45 on rotated parity), because most of the work is per node, not per sample.
  On vehicle, sampling projections takes 49% of a LOT fit, building oblique columns 28%, and the gain maths only 7%.

![fit time per dataset and as N grows](results/figures/timing.png)

### Ablations

![ablations](results/figures/ablations.png)

LOT on rotated parity unless noted, test error:

- (a) Features allowed per split, by level: 3 everywhere 1.9%, fewer near the leaves 2.5%, fewer near the root 7.8%.
- (b) Looking ahead only at the top 10 candidates by gain: 7.4% instead of 1.9%. The right first split has zero gain, so it gets cut.
- (c) Parity of order 3 with Look axis: 6.5% at depth 3, 2.4% at depth 4.
- (d) Axis thresholds built per node, like the oblique ones: vehicle's oblique gain drops from 3.5 to 1.5 points,
  so part of the gain came from the per-node thresholds, not from the oblique direction.

## References

1. SPLIT: *Near-Optimal Decision Trees in a SPLIT Second*, ICML 2025.
2. SPORF: Tomita et al., *Sparse Projection Oblique Randomer Forests*, JMLR 2020.
