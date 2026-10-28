import csv
import json
import os
import subprocess

"""download the real datasets of spec 8.2 and write datasets/<name>.csv: header, features, class 0..C-1 last
curl, not urllib: Python's TLS handshake with archive.ics.uci.edu times out where curl connects"""
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
raw = os.path.join(root, "raw")


def fetch(url, name):
    path = os.path.join(raw, name)
    if not os.path.exists(path):
        print("downloading", url)
        subprocess.run(["curl", "-sSfL", "--retry", "3", "--max-time", "120", "--ssl-no-revoke", "-o", path, url], check=True)
    return open(path, newline="").read()


def write(name, header, X, labels):
    cls = sorted(set(labels))
    with open(os.path.join(root, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["class"])
        for x, c in zip(X, labels):
            w.writerow(x + [cls.index(c)])
    print(name, len(X), "rows,", len(header), "features, classes", cls)


def uciApi(uid, name):
    """the UCI API names each column's role; ID columns are dropped, the Target becomes the class"""
    meta = json.loads(fetch("https://archive.ics.uci.edu/api/dataset?id=" + str(uid), name + ".json"))["data"]
    role = {}
    for v in meta["variables"]:
        role[v["name"]] = v["role"]
    rows = list(csv.reader(fetch(meta["data_url"], name + ".data.csv").splitlines()))
    feat = [j for j in range(len(rows[0])) if role[rows[0][j]] == "Feature"]
    target = [j for j in range(len(rows[0])) if role[rows[0][j]] == "Target"][0]
    X = [[r[j] for j in feat] for r in rows[1:]]
    write(name, [rows[0][j] for j in feat], X, [r[target] for r in rows[1:]])


def vehicle():
    """the original statlog files: the API's copy has one row shifted by a column, losing its class"""
    X = []
    labels = []
    for part in "abcdefghi":
        name = "xa" + part + ".dat"
        for line in fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/" + name, name).splitlines():
            v = line.split()
            if v:
                X.append(v[:18])
                labels.append(v[18])
    write("vehicle", ["f" + str(j + 1) for j in range(18)], X, labels)


def compas():
    """ProPublica's filter (6172 rows); the first 5 features are numeric, the other 6 binary"""
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    races = ["African-American", "Caucasian", "Hispanic"]
    X = []
    labels = []
    for r in csv.DictReader(fetch(url, "compas-scores-two-years.csv").splitlines()):
        if r["days_b_screening_arrest"] == "" or abs(int(r["days_b_screening_arrest"])) > 30:
            continue
        if r["is_recid"] == "-1" or r["c_charge_degree"] == "O" or r["score_text"] == "N/A":
            continue
        x = [r["age"], r["priors_count"], r["juv_fel_count"], r["juv_misd_count"], r["juv_other_count"],
             int(r["sex"] == "Male"), int(r["c_charge_degree"] == "F")]
        for race in races:
            x.append(int(r["race"] == race))
        x.append(int(r["race"] not in races))
        X.append(x)
        labels.append(r["two_year_recid"])
    header = ["age", "priors", "juvFel", "juvMisd", "juvOther", "male", "felony",
              "raceAfricanAmerican", "raceCaucasian", "raceHispanic", "raceOther"]
    write("compas", header, X, labels)


os.makedirs(raw, exist_ok=True)
uciApi(12, "balance_scale")
uciApi(17, "wdbc")
uciApi(94, "spambase")
uciApi(159, "magic")
uciApi(109, "wine")
vehicle()
compas()
