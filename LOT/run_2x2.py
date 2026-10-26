import os

"""one BLAS thread per process: the pool already uses every core"""
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import sys
import time
from multiprocessing import Pool
import numpy as np
from lot.data import parity, rotatedParity, diagonal, orthant, trunk, ringnorm, loadCsv, subsample, stratifiedFolds
from lot.tree import LOT
from lot.metrics import kappa, countNnz, countExtra

"""spec 8.2: the four variants on every dataset, 5-fold stratified CV repeated with 3 seeds, every lambda
of the grid with gamma = lam/2, plus gamma = 0 at the headline lambda; one row per fit in results/runs.csv"""
here = os.path.dirname(os.path.abspath(__file__))
outPath = os.path.join(here, "results", "runs.csv")
lams = (0.001, 0.005, 0.01, 0.02)
headline = 0.005
seeds = (0, 1, 2)
nFolds = 5
variants = [(False, False), (True, False), (False, True), (True, True)]
header = ["dataset", "seed", "fold", "lam", "gamma", "L", "O", "testError", "kappa", "leaves", "nnz", "extra",
          "fitSeconds", "trainObjective"]


def datasets():
    """synthetic: n = 1000 with 2 noise features (spec 8.2 leaves p_noise open); real: N <= 2000
    the value is (X, y, numeric): COMPAS projects its 5 numeric features only (spec 9)"""
    real = os.path.join(here, "datasets")
    compas = subsample(*loadCsv(os.path.join(real, "compas.csv")), 2000, 0)
    return {
        "parity2": parity(1000, 2, 2, 0) + (None,),
        "parity3": parity(1000, 3, 2, 0) + (None,),
        "rotatedParity": rotatedParity(1000, 2, 0) + (None,),
        "diagonal": diagonal(1000, 2, 0) + (None,),
        "orthant": orthant(1000, 3, 2, 0) + (None,),
        "trunk": trunk(1000, 10, 0) + (None,),
        "ringnorm": ringnorm(2000, 0) + (None,),
        "balanceScale": loadCsv(os.path.join(real, "balance_scale.csv")) + (None,),
        "vehicle": loadCsv(os.path.join(real, "vehicle.csv")) + (None,),
        "wdbc": loadCsv(os.path.join(real, "wdbc.csv")) + (None,),
        "compas": compas + ([0, 1, 2, 3, 4],),
    }


data = None


def runJob(job):
    global data
    if data is None:
        data = datasets()
    name, seed, fold, lam, gamma = job
    X, y, numeric = data[name]
    test = stratifiedFolds(y, nFolds, seed) == fold
    rows = []
    cost = {}
    for la, ob in variants:
        t = time.time()
        m = LOT(lam=lam, gamma=gamma, lookahead=la, oblique=ob, seed=seed, numeric=numeric).fit(X[~test], y[~test])
        dt = time.time() - t
        pred = m.predict(X[test])
        cost[(la, ob)] = m.cost
        rows.append([name, seed, fold, lam, gamma, int(la), int(ob), np.mean(pred != y[test]), kappa(y[test], pred),
                     m.nLeaves(), countNnz(m.tree), countExtra(m.tree), round(dt, 3), m.cost])
    for ob in (False, True):
        """Proposition 12 on every run (spec 7.8 item 5)"""
        assert cost[(True, ob)] <= cost[(False, ob)] + 1e-12, job
    return rows


def allJobs():
    jobs = []
    for name in datasets():
        for seed in seeds:
            for fold in range(nFolds):
                for lam in lams:
                    jobs.append((name, seed, fold, lam, lam / 2))
                jobs.append((name, seed, fold, headline, 0.0))
    return jobs


def doneJobs():
    done = set()
    if os.path.exists(outPath):
        for r in csv.DictReader(open(outPath)):
            done.add((r["dataset"], int(r["seed"]), int(r["fold"]), float(r["lam"]), float(r["gamma"])))
    return done


if __name__ == "__main__":
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    done = doneJobs()
    jobs = [j for j in allJobs() if j not in done]
    print(len(jobs), "jobs to run,", len(done), "already done", flush=True)
    fresh = not os.path.exists(outPath)
    out = open(outPath, "a", newline="")
    w = csv.writer(out)
    if fresh:
        w.writerow(header)
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() - 1) as pool:
        for k, rows in enumerate(pool.imap_unordered(runJob, jobs)):
            w.writerows(rows)
            out.flush()
            print(k + 1, "/", len(jobs), rows[0][0], "%.0fs" % (time.time() - t0), flush=True)
