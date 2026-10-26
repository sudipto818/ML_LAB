# LOT: Lookahead Oblique Trees

LOT is a decision-tree learner that combines two ideas from the two papers: the lookahead scoring of **SPLIT** (*Near-Optimal Decision Trees in a SPLIT Second*, ICML 2025) and the sparse oblique splits of **SPORF** (*Sparse Projection Oblique Randomer Forests*, JMLR 2020). It is written in Python, as a project for DA322 (Machine Learning), IIT Guwahati.

## Idea

- A greedy tree picks each split by information gain alone, so it fails on problems like XOR where no single test shows any gain.
- SPLIT-style lookahead scores a candidate split by the cost of the best small tree below it, so a split with zero immediate gain can still be chosen.
- SPORF-style oblique splits test sparse combinations $a^\top x \le \tau$ with $a \in \{-1,0,+1\}^p$, which capture diagonal boundaries that axis-aligned trees need many leaves to approximate.
- LOT puts both in one recursion. It minimises

  $$R(T)=\frac{1}{N}\sum_{i}\mathbf 1[T(x_i)\neq y_i]+\lambda\,S(T)+\gamma\sum_{v}c(a_v),$$

  where $S(T)$ is the number of leaves and $c(a)=\|a\|_0-1$ penalises dense projections.

Two switches give four variants: greedy or lookahead, and axis-only or axis plus oblique. The lookahead and oblique variant is LOT.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          
pip install -r requirements.txt
```

## Run the tests

From inside `LOT/`:

```bash
python -m pytest
```

## Run the experiments

From inside `LOT/`, in this order:

```bash
python fetch_data.py               # balance-scale, wdbc (UCI API), vehicle (statlog files), COMPAS; needs curl
python run_2x2.py                  # 4 variants x 11 datasets x 5 folds x 3 seeds x lambda grid -> results/runs.csv
python ablations.py                # schedule, screening, depth, threshold control -> results/ablations.csv
python make_plots.py               # results/headline.md, headline.csv and the figures
python -m demos.demo_xor           # one script per concept demonstration (spec Table 2)
```

`run_2x2.py` and `ablations.py` use every core but one and resume where they stopped if interrupted.

## Layout

| Folder | Contents |
|---|---|
| `lot/` | the method: data generators, thresholds, projections, tree, metrics |
| `tests/` | unit tests, including checks that lookahead never does worse than greedy on the training objective |
| `demos/` | one script per concept: XOR, diagonal, rotated parity, sparse vs dense, projection count |
| `results/` | `runs.csv`, `ablations.csv`, `headline.md`, `figures/` |
| `datasets/` | downloaded by `fetch_data.py`, not committed |

## Status

- [x] Data generators, CSV loading, stratified folds
- [x] Scoring core (entropy, gains, leaf cost)
- [x] Axis-only greedy and lookahead
- [x] Oblique candidates, schedules s(level), feature penalty gamma
- [x] Concept demonstrations (spec Table 2)
- [x] 2 x 2 study and ablations (spec 8.2, 8.4)
- [ ] Report and presentation

## References

1. SPLIT: *Near-Optimal Decision Trees in a SPLIT Second*, ICML 2025.
2. SPORF: Tomita et al., *Sparse Projection Oblique Randomer Forests*, JMLR 2020.

The papers are in `../Papers/`.