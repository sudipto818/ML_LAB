import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lot.oblique import nProjections
from experiment import datasets, headline, lams

"""results/runs.csv and results/ablations.csv -> headline table, interaction I, coverage pi_r, figures
the 2 x 2 is drawn as colour = candidates (blue axis, orange oblique), fill = selection (hollow Greedy, filled Look)"""
here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "results")
blue, orange, ink2, muted = "#2a78d6", "#eb6834", "#52514e", "#898781"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "text.color": "#0b0b0b", "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    path = os.path.join(res, "figures", name)
    fig.savefig(path)
    print("saved", path)
ds = datasets()
names = list(ds)
nums = {}
for k, v in ds.items():
    nums[k] = v[0].shape[1] if v[2] is None else len(v[2])
variantName = {(0, 0): "Greedy axis", (1, 0): "Look axis", (0, 1): "Greedy oblique", (1, 1): "LOT"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("seed", "fold", "L", "O", "leaves", "nnz", "extra"):
            r[k] = int(r[k])
        for k in ("lam", "testError", "kappa", "fitSeconds", "trainObjective"):
            if k in r:
                r[k] = float(r[k])
        if "gamma" in r:
            r["gamma"] = float(r["gamma"])
    return rows


def sel(rows, **cond):
    return [r for r in rows if all(r[k] == v for k, v in cond.items())]


def ms(xs):
    return np.mean(xs), np.std(xs)


def marker(ax, x, yv, L, O, err=None, label=None):
    c = orange if O else blue
    ax.errorbar(x, yv, xerr=err, fmt="o", ms=6, color=c, mfc=c if L else "white", mec=c, mew=1.4, capsize=2,
                label=label)


runs = load(os.path.join(res, "runs.csv"))
head = [r for r in runs if r["lam"] == headline and abs(r["gamma"] - headline / 2) < 1e-12]

"""headline table and interaction I = (E00 - E10) - (E01 - E11), computed per (seed, fold), then mean, sd"""
table = []
for name in names:
    rs = sel(head, dataset=name)
    E = {}
    for r in rs:
        E[(r["seed"], r["fold"], r["L"], r["O"])] = r["testError"]
    I = []
    for seed, fold in sorted({(r["seed"], r["fold"]) for r in rs}):
        e = lambda L, O: E[(seed, fold, L, O)]
        I.append((e(0, 0) - e(1, 0)) - (e(0, 1) - e(1, 1)))
    p = nums[name]
    row = {"dataset": name, "p": p, "pi2": min(1, 30 / nProjections(p, 2)),
           "pi3": min(1, 30 / nProjections(p, 3)) if p >= 3 else float("nan"), "I": ms(I), "runs": len(I)}
    for (L, O), vn in variantName.items():
        v = sel(rs, L=L, O=O)
        row[vn] = {k: ms([r[k] for r in v]) for k in ("testError", "kappa", "leaves", "nnz", "fitSeconds")}
    table.append(row)

with open(os.path.join(res, "headline.csv"), "w", newline="") as f:
    w = csv.writer(f)
    cols = ["testError", "kappa", "leaves", "nnz", "fitSeconds"]
    w.writerow(["dataset", "p", "pi2", "pi3", "I_mean", "I_sd", "runs"] +
               [vn + " " + c + s for vn in variantName.values() for c in cols for s in (" mean", " sd")])
    for row in table:
        vals = [row["dataset"], row["p"], round(row["pi2"], 4), round(row["pi3"], 4), round(row["I"][0], 4),
                round(row["I"][1], 4), row["runs"]]
        for vn in variantName.values():
            for c in cols:
                vals += [round(row[vn][c][0], 4), round(row[vn][c][1], 4)]
        w.writerow(vals)

with open(os.path.join(res, "headline.md"), "w") as f:
    f.write("Test error, mean +- sd over %d runs (5-fold CV x 3 seeds), lambda = %g, gamma = lambda/2\n\n"
            % (table[0]["runs"], headline))
    f.write("| dataset | p | pi2 | " + " | ".join(variantName.values()) + " | I | LOT leaves | LOT nnz |\n")
    f.write("|---" * 10 + "|\n")
    for row in table:
        cells = ["%.3f +- %.3f" % row[vn]["testError"] for vn in variantName.values()]
        f.write("| %s | %d | %.2f | %s | %+.3f +- %.3f | %.1f | %.1f |\n" % (
            row["dataset"], row["p"], row["pi2"], " | ".join(cells), row["I"][0], row["I"][1],
            row["LOT"]["leaves"][0], row["LOT"]["nnz"][0]))
    f.write("\nCohen's kappa, mean +- sd\n\n| dataset | " + " | ".join(variantName.values()) + " |\n")
    f.write("|---" * 5 + "|\n")
    for row in table:
        f.write("| %s | %s |\n" % (row["dataset"], " | ".join("%.3f +- %.3f" % row[vn]["kappa"]
                                                            for vn in variantName.values())))
    f.write("\nFit time in seconds, mean\n\n| dataset | " + " | ".join(variantName.values()) + " |\n")
    f.write("|---" * 5 + "|\n")
    for row in table:
        f.write("| %s | %s |\n" % (row["dataset"], " | ".join("%.3f" % row[vn]["fitSeconds"][0]
                                                            for vn in variantName.values())))
    f.write("\nFeature penalty at lambda = %g: gamma = 0 against gamma = lambda/2, oblique variants\n\n" % headline)
    f.write("| dataset | Greedy oblique error g=0 | g=lam/2 | nnz g=0 | g=lam/2 | LOT error g=0 | g=lam/2 | nnz g=0 | g=lam/2 |\n")
    f.write("|---" * 9 + "|\n")
    for name in names:
        cells = []
        for L in (0, 1):
            a = sel(runs, dataset=name, lam=headline, gamma=0.0, L=L, O=1)
            b = sel(head, dataset=name, L=L, O=1)
            cells += ["%.3f" % np.mean([r["testError"] for r in a]), "%.3f" % np.mean([r["testError"] for r in b]),
                      "%.1f" % np.mean([r["nnz"] for r in a]), "%.1f" % np.mean([r["nnz"] for r in b])]
        f.write("| %s | %s |\n" % (name, " | ".join(cells)))
print(open(os.path.join(res, "headline.md")).read())

"""figure: test error of the four variants and the interaction I, one row per dataset"""
order = names[::-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
for i, name in enumerate(order):
    row = [r for r in table if r["dataset"] == name][0]
    for k, ((L, O), vn) in enumerate(variantName.items()):
        mu, sd = row[vn]["testError"]
        marker(a, mu, i + (1.5 - k) * 0.16, L, O, sd, vn if i == 0 else None)
    mu, sd = row["I"]
    b.errorbar(mu, i, xerr=sd, fmt="o", ms=6, color=ink2, capsize=2)
"""synthetic above the line, real data below"""
cut = order.index("ringnorm") - 0.5
for ax in (a, b):
    ax.axhline(cut, color=ink2, lw=0.8)
a.text(0.99, cut + 0.12, "synthetic", transform=a.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=ink2)
a.text(0.99, cut - 0.12, "real", transform=a.get_yaxis_transform(), ha="right", va="top", fontsize=8, color=ink2)
a.set_yticks(range(len(order)), order)
a.set(xlabel="test error (mean, sd over 15 runs)", title=r"(a) Four variants, $\lambda$ = %g" % headline, xlim=(-0.01, None))
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
b.axvline(0, color=muted, lw=1)
b.xaxis.set_major_locator(plt.MaxNLocator(5))
b.set(xlabel=r"$I = (E_{00}-E_{10}) - (E_{01}-E_{11})$", title="(b) Interaction")
b.text(0.02, 0.01, "I < 0:\ncomplements", transform=b.transAxes, ha="left", va="bottom", fontsize=8, color=ink2)
b.text(0.98, 0.01, "I > 0:\nsubstitutes", transform=b.transAxes, ha="right", va="bottom", fontsize=8, color=ink2)
fig.tight_layout()
save(fig, "headline.png")

"""figure: the lambda sweep, test error against leaves, one panel per dataset (Figure 4 of SPLIT)"""
fig, axs = plt.subplots(3, 4, figsize=(12, 8.4))
for ax, name in zip(axs.flat, names):
    for (L, O), vn in variantName.items():
        pts = []
        for lam in lams:
            v = sel(runs, dataset=name, lam=lam, L=L, O=O)
            v = [r for r in v if abs(r["gamma"] - lam / 2) < 1e-12]
            pts.append((np.mean([r["leaves"] for r in v]), np.mean([r["testError"] for r in v])))
        c = orange if O else blue
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=c, lw=1.2, ls="-" if L else (0, (3, 2)))
        for q in pts:
            marker(ax, q[0], q[1], L, O)
    ax.set_title(name)
    ax.set_xscale("log", base=2)
    lo, hi = ax.get_ylim()
    if hi - lo < 0.02:
        ax.set_ylim((lo + hi) / 2 - 0.01, (lo + hi) / 2 + 0.01)
for ax in axs[-1]:
    ax.set_xlabel("leaves (mean)")
for ax in axs[:, 0]:
    ax.set_ylabel("test error (mean)")
last = axs.flat[-1]
last.axis("off")
for (L, O), vn in variantName.items():
    marker(last, np.nan, np.nan, L, O, label=vn)
last.legend(loc="center", title=r"one point per $\lambda \in \{%s\}$" % ", ".join("%g" % l for l in lams))
last.text(0.5, 0.12, "a hidden marker lies under LOT:\nthose variants found the same trees", transform=last.transAxes,
          ha="center", fontsize=8, color=ink2)
fig.tight_layout()
save(fig, "lambda.png")

"""figure: ablations"""
abl = load(os.path.join(res, "ablations.csv"))
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for k, (ablation, ax, title) in enumerate([("schedule", axs[0], "(a) Schedule, rotated parity (H5)"),
                                           ("screening", axs[1], "(b) Gain screening, rotated parity (H6)")]):
    settings = list(dict.fromkeys(r["setting"] for r in sel(abl, ablation=ablation)))
    for i, s in enumerate(settings):
        v = sel(abl, ablation=ablation, setting=s)
        mu, sd = ms([r["testError"] for r in v])
        marker(ax, i, mu, 1, 1)
        ax.errorbar(i, mu, yerr=sd, color=orange, capsize=3)
        ax.annotate("nnz %.1f" % np.mean([r["nnz"] for r in v]), (i, mu), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, color=ink2, va="center")
    ax.set_xticks(range(len(settings)), settings)
    ax.set_xlim(-0.5, len(settings) - 0.3)
    ax.set(ylabel="LOT test error (mean, sd)", title=title, ylim=(-0.02, None))
ax = axs[2]
for i, d in enumerate(("3", "4")):
    for L in (0, 1):
        v = sel(abl, ablation="depth", setting=d, L=L)
        mu, sd = ms([r["testError"] for r in v])
        marker(ax, i + (L - 0.5) * 0.2, mu, L, 0, label=variantName[(L, 0)] if i == 0 else None)
        ax.errorbar(i + (L - 0.5) * 0.2, mu, yerr=sd, color=blue, capsize=3)
ax.set_xticks([0, 1], ["depth 3", "depth 4"])
ax.set_xlim(-0.5, 1.5)
ax.set(ylabel="test error (mean, sd)", title="(c) Parity of order 3 (H7)", ylim=(-0.02, None))
ax.legend(loc="upper right")
ax = axs[3]
for i, name in enumerate(order):
    base = sel(head, dataset=name)
    ctrl = sel(abl, ablation="thresholds", dataset=name)
    gain = lambda rs, L: np.mean([r["testError"] for r in rs if r["L"] == L and r["O"] == 0]) - \
        np.mean([r["testError"] for r in rs if r["L"] == L and r["O"] == 1])
    ax.plot([gain(base, 1), gain(ctrl, 1)], [i, i], color=muted, lw=1, zorder=0)
    ax.plot(gain(base, 1), i, "o", ms=6, color=orange, label="global axis thresholds" if i == 0 else None)
    ax.plot(gain(ctrl, 1), i, "D", ms=5, color=ink2, label="node axis thresholds" if i == 0 else None)
ax.axvline(0, color=muted, lw=1)
ax.set_yticks(range(len(order)), order)
ax.set(xlabel=r"oblique benefit under Look, $E_{10} - E_{11}$", title="(d) Threshold control")
ax.legend(loc="lower right", fontsize=7)
fig.tight_layout()
save(fig, "ablations.png")
