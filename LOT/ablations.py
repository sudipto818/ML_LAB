from runner import doneRows, runAll
import os
import time
import numpy as np
from lot.data import stratifiedFolds
from lot.tree import LOT, schedules
from lot.metrics import kappa, countNnz, countExtra
from experiment import datasets, seeds, nFolds, headline, variants

"""spec 8.4 and H5-H7 at the headline lambda, gamma = lam/2, same folds as experiment.py; the gamma
ablation is already in runs.csv (gamma = 0 rows). One row per fit in results/ablations.csv"""
here = os.path.dirname(os.path.abspath(__file__))
outPath = os.path.join(here, "results", "ablations.csv")
header = ["ablation", "setting", "dataset", "seed", "fold", "L", "O", "testError", "kappa", "leaves", "nnz", "extra",
          "fitSeconds", "trainObjective"]
lot = [(True, True)]
settings = [
    ("schedule", "flat", ["rotatedParity"], lot, {"s": schedules["flat"]}),
    ("schedule", "rootHeavy", ["rotatedParity"], lot, {"s": schedules["rootHeavy"]}),
    ("schedule", "tailHeavy", ["rotatedParity"], lot, {"s": schedules["tailHeavy"]}),
    ("screening", "off", ["rotatedParity"], lot, {}),
    ("screening", "top10", ["rotatedParity"], lot, {"screen": 10}),
    ("depth", "3", ["parity3"], [(False, False), (True, False)], {"depth": 3}),
    ("depth", "4", ["parity3"], [(False, False), (True, False)], {"depth": 4}),
]
for name in datasets():
    settings.append(("thresholds", "nodeAxis", [name], variants, {"axisLocal": True}))

data = None


def runJob(job):
    global data
    if data is None:
        data = datasets()
    ablation, setting, name, seed, fold = job[:5]
    vs, kw = job[5], job[6]
    X, y, numeric = data[name]
    test = stratifiedFolds(y, nFolds, seed) == fold
    rows = []
    for la, ob in vs:
        t = time.time()
        m = LOT(lam=headline, gamma=headline / 2, lookahead=la, oblique=ob, seed=seed, numeric=numeric, **kw)
        m.fit(X[~test], y[~test])
        dt = time.time() - t
        pred = m.predict(X[test])
        rows.append([ablation, setting, name, seed, fold, int(la), int(ob), np.mean(pred != y[test]),
                     kappa(y[test], pred), m.nLeaves(), countNnz(m.tree), countExtra(m.tree), round(dt, 3), m.cost])
    return rows


if __name__ == "__main__":
    done = set()
    for r in doneRows(outPath):
        done.add((r["ablation"], r["setting"], r["dataset"], int(r["seed"]), int(r["fold"])))
    jobs = []
    for ablation, setting, names, vs, kw in settings:
        for name in names:
            for seed in seeds:
                for fold in range(nFolds):
                    if (ablation, setting, name, seed, fold) not in done:
                        jobs.append((ablation, setting, name, seed, fold, vs, kw))
    runAll(outPath, header, jobs, runJob)
