import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""series colours in fixed order, then greys"""
blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
ink, ink2, muted, grid, axis = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
blueTint, orangeTint, wrong = "#dbe8f8", "#fbe3d9", "#6f6c66"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "text.color": ink, "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
    "axes.edgecolor": axis, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
})

outDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "figures")


def save(fig, name):
    os.makedirs(outDir, exist_ok=True)
    path = os.path.normpath(os.path.join(outDir, name))
    fig.savefig(path)
    print("saved", path)


def scatterClasses(ax, X, y, i, j):
    """marker shape encodes the class too, not only colour"""
    ax.scatter(X[y == 0, i], X[y == 0, j], s=9, c=blue, marker="o", linewidths=0, label="class 0")
    ax.scatter(X[y == 1, i], X[y == 1, j], s=12, c=orange, marker="x", linewidths=0.9, label="class 1")
