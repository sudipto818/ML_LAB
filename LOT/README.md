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
python -m pytest                     # tests
python fetch.py                      # download the real datasets (needs curl)
python experiment.py                 # main experiment -> results/runs.csv
python ablations.py                  # ablations -> results/ablations.csv
python plots.py                      # tables and figures
python -m demos.demo_rotated_parity  # small demo, about 2 seconds
```

## Folders

- `lot/` the tree, data, projections, metrics
- `tests/` unit tests
- `demos/` small demos
- `results/` csv files and figures

## References

1. SPLIT: *Near-Optimal Decision Trees in a SPLIT Second*, ICML 2025.
2. SPORF: Tomita et al., *Sparse Projection Oblique Randomer Forests*, JMLR 2020.
