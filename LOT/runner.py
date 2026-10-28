import os

"""one BLAS thread per process: the pool already uses every core; import this before numpy"""
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import sys
import time
from multiprocessing import Pool


def doneRows(outPath):
    if not os.path.exists(outPath):
        return []
    return list(csv.DictReader(open(outPath)))


def runAll(outPath, header, jobs, runJob):
    """runJob(job) -> rows, appended to outPath as they finish, so a rerun resumes"""
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    print(len(jobs), "jobs to run", flush=True)
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
            print(k + 1, "/", len(jobs), rows[0][:3], "%.0fs" % (time.time() - t0), flush=True)
