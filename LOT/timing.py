import csv
import os
import platform
import subprocess
import time
import numpy as np
from lot.data import rotatedParity, stratifiedFolds
from lot.tree import LOT
from experiment import datasets, headline, variants

"""fit time of the 4 variants, one process, one fit at a time: every dataset on the 5 folds of seed 0,
and rotated parity as N grows; writes results/timing.csv and results/timing.md"""
here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "results")
names = {(False, False): "Greedy axis", (True, False): "Look axis", (False, True): "Greedy oblique", (True, True): "LOT"}
sizes = (250, 500, 1000, 2000, 4000)


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def device():
    perf = sh("sysctl", "-n", "hw.perflevel0.physicalcpu")
    eff = sh("sysctl", "-n", "hw.perflevel1.physicalcpu")
    ram = int(sh("sysctl", "-n", "hw.memsize")) // 2 ** 30
    power = sh("pmset", "-g", "batt").splitlines()[0].split("'")[1]
    return ["Machine: " + sh("sysctl", "-n", "hw.model") + ", " + sh("sysctl", "-n", "machdep.cpu.brand_string")
            + ", " + perf + " performance + " + eff + " efficiency cores, " + str(ram) + " GB RAM",
            "OS: macOS " + platform.mac_ver()[0] + "; Python " + platform.python_version() + ", NumPy " + np.__version__,
            "Power: " + power + "; one fit at a time, nothing else running"]


def fitSeconds(X, y, la, ob, numeric=None):
    t = time.perf_counter()
    LOT(lam=headline, gamma=headline / 2, lookahead=la, oblique=ob, numeric=numeric).fit(X, y)
    return time.perf_counter() - t


def row(part, name, X, variant, seconds):
    return {"part": part, "dataset": name, "n": len(X), "p": X.shape[1], "variant": variant, "seconds": seconds}


def median(rows, **cond):
    return np.median([r["seconds"] for r in rows if all(r[k] == v for k, v in cond.items())])


if __name__ == "__main__":
    spec = device()
    fitSeconds(*rotatedParity(300, 2, 0), True, True)
    rows = []
    for name, (X, y, numeric) in datasets().items():
        fold = stratifiedFolds(y, 5, 0)
        for f in range(5):
            Xf, yf = X[fold != f], y[fold != f]
            for la, ob in variants:
                rows.append(row("dataset", name, Xf, names[(la, ob)], fitSeconds(Xf, yf, la, ob, numeric)))
        print(name, "%.1fs" % median(rows, dataset=name, variant="LOT"), flush=True)
    for n in sizes:
        X, y = rotatedParity(n, 2, 0)
        for la, ob in variants:
            rows.append(row("size", "rotatedParity", X, names[(la, ob)], fitSeconds(X, y, la, ob)))
        print("N =", n, "%.1fs" % rows[-1]["seconds"], flush=True)

    with open(os.path.join(res, "timing.csv"), "w", newline="") as out:
        w = csv.DictWriter(out, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Timing", "", "Exact wall-clock fit times, measured once with `python timing.py`.", ""]
    md += ["- " + s for s in spec]
    md += ["- depth 4, lambda = " + str(headline) + ", gamma = lambda/2, the settings of the headline table", ""]
    md += ["## Fit time per dataset (s, median of the 5 training folds of seed 0)", ""]
    md += ["| Dataset | N train | p | " + " | ".join(names.values()) + " | LOT / Look axis |", "|---|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(r["dataset"] for r in rows if r["part"] == "dataset"):
        n = [r for r in rows if r["dataset"] == name][0]
        t = [median(rows, part="dataset", dataset=name, variant=v) for v in names.values()]
        md.append("| %s | %d | %d | %s | %.0fx |" % (name, n["n"], n["p"], " | ".join("%.3f" % s for s in t), t[3] / t[1]))
    md += ["", "## Fit time as N grows (s, rotated parity with 2 noise features, p = 6)", ""]
    md += ["| N | " + " | ".join(names.values()) + " |", "|---|---|---|---|---|"]
    for n in sizes:
        md.append("| %d | %s |" % (n, " | ".join("%.3f" % median(rows, part="size", n=n, variant=v) for v in names.values())))
    """growth exponent b of t = a N^b, a least-squares line on log t against log N"""
    b = [np.polyfit(np.log(sizes), np.log([median(rows, part="size", n=n, variant=v) for n in sizes]), 1)[0]
         for v in names.values()]
    md.append("| exponent b in t ~ N^b | %s |" % " | ".join("%.2f" % x for x in b))
    open(os.path.join(res, "timing.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
