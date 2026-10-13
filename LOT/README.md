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

## Layout

| Folder | Contents |
|---|---|
| `lot/` | the method: data generators, thresholds, projections, tree, metrics |
| `tests/` | unit tests, including checks that lookahead never does worse than greedy on the training objective |
| `demos/` | XOR, diagonal boundary and rotated parity examples |
| `results/` | experiment outputs |

## Status

- [ ] Data generators
- [ ] Scoring core (entropy, gains, leaf cost)
- [ ] Axis-only greedy and lookahead
- [ ] Oblique candidates
- [ ] Experiments and report

## References

1. SPLIT: *Near-Optimal Decision Trees in a SPLIT Second*, ICML 2025.
2. SPORF: Tomita et al., *Sparse Projection Oblique Randomer Forests*, JMLR 2020.

The papers are in `../Papers/`.