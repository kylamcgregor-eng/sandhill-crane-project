"""
08_concept_diagrams2.py  -  Explanatory diagrams for the Naive Bayes and Decision Trees Overview
sections (Module 3). Illustrate the methods with small synthetic examples, not the project's crane
data (the Results sections on each tab use the real data). Each panel says so in its own title.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
NDIR = C.FIGS.parent / "naivebayes"
DDIR = C.FIGS.parent / "dectrees"
NDIR.mkdir(parents=True, exist_ok=True)
DDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight",
})


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


# ------------------------------------------------------------------ 1. NB: the naive independence assumption
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.add_patch(plt.Circle((5, 5), 0.75, facecolor="#ffe3d1", edgecolor=INK2, linewidth=1.3))
ax.text(5, 5, "class y", ha="center", va="center", fontsize=10.5, fontweight="bold")
xs = [1.5, 4, 6, 8.5]
labels = ["location\n(lat, lon)", "flock size\n(count)", "day of\nyear", "..."]
for x, lab in zip(xs, labels):
    ax.add_patch(plt.Circle((x, 1.4), 0.75, facecolor="#cde2fb", edgecolor=INK2, linewidth=1.3))
    ax.text(x, 1.4, lab, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(x, 2.15), xytext=(5 - (5 - x) * 0.55, 4.25),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
ax.text(5, 0.25, "No arrows run between the features themselves - each one is assumed independent of the\n"
                 "others once the class is known. That assumption is what makes Naive Bayes fast to fit.",
        ha="center", va="center", fontsize=10, color=INK2)
ax.set_title("The naive assumption: features are conditionally independent given the class\n(illustrative example)",
             fontsize=12.5, loc="left")
save(fig, NDIR, "concept_naive_independence")

# ------------------------------------------------------------------ 2. NB: multinomial vs. Bernoulli features
rng = np.random.default_rng(5612)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
words = ["north", "south", "flock", "pair"]
counts = [5, 1, 3, 0]
ax.bar(words, counts, color=BLUE)
ax.set_title("Multinomial NB: feature is a count\n(how many times, e.g. word frequency)", fontsize=11)
ax.set_ylabel("count in this example")
ax = axes[1]
present = [1, 1, 1, 0]
ax.bar(words, present, color=ORANGE)
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 1]); ax.set_yticklabels(["absent", "present"])
ax.set_title("Bernoulli NB: feature is binary\n(did it occur at all, yes/no)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.82])
fig.suptitle("Two ways Naive Bayes can model a feature (illustrative example, not project data)",
             x=0.01, y=0.98, ha="left", va="top", fontweight="bold", fontsize=13)
save(fig, NDIR, "concept_multinomial_vs_bernoulli")

# ------------------------------------------------------------------ 3. DT: why (close to) infinitely many trees exist
rng = np.random.default_rng(11)
cls_a = rng.normal((2, 2), 0.7, size=(18, 2))
cls_b = rng.normal((5, 5), 0.7, size=(18, 2))
pts = np.vstack([cls_a, cls_b])
lab = np.array([0] * 18 + [1] * 18)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
cuts = [
    ("Tree A: one cut on x", [("v", 3.5)]),
    ("Tree B: one cut on y", [("h", 3.5)]),
    ("Tree C: two cuts, different thresholds", [("v", 3.0), ("h", 4.2)]),
]
for ax, (title, lines) in zip(axes, cuts):
    for c, col in ((0, BLUE), (1, ORANGE)):
        m = lab == c
        ax.scatter(pts[m, 0], pts[m, 1], s=28, color=col, alpha=0.85)
    for kind, v in lines:
        if kind == "v":
            ax.axvline(v, color=INK, lw=1.8)
        else:
            ax.axhline(v, color=INK, lw=1.8)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(rect=[0, 0, 1, 0.80])
fig.suptitle("Any of infinitely many thresholds could be the cut - a slightly different threshold, a\n"
             "different feature, or a different depth all produce a different, equally valid tree (illustrative example)",
             x=0.01, y=0.98, ha="left", va="top", fontweight="bold", fontsize=12.5)
save(fig, DDIR, "concept_infinite_trees")

# ------------------------------------------------------------------ 4. DT: impurity before/after a split
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
ax = axes[0]
parent = [6, 6]
ax.pie(parent, colors=[BLUE, ORANGE], labels=["class A\n(6)", "class B\n(6)"],
       autopct=None, startangle=90, wedgeprops=dict(edgecolor=SURFACE, linewidth=2))
ax.set_title("Before the split\nGini = 1 - (0.5²+0.5²) = 0.50\n(worst possible mix for two classes)", fontsize=10.5)
ax = axes[1]
left = [5, 1]
right = [1, 5]
w, h = 0.42, 0.9
ax.axis("off")
for cx, counts, name in ((0.22, left, "left child"), (0.72, right, "right child")):
    total = sum(counts)
    yb = 0.05
    for val, col in zip(counts, (BLUE, ORANGE)):
        hh = h * val / total
        ax.add_patch(plt.Rectangle((cx - w / 2, yb), w, hh, facecolor=col, edgecolor=SURFACE))
        yb += hh
    g = 1 - sum((v / total) ** 2 for v in counts)
    ax.text(cx, -0.08, f"{name}\n(n={total})\nGini={g:.2f}", ha="center", va="top", fontsize=9.5, transform=ax.transAxes)
ax.set_xlim(0, 1); ax.set_ylim(-0.25, 1.05)
ax.set_title("After a good split\nweighted Gini ≈ 0.5·(0.28) + 0.5·(0.28) = 0.28\nGini gain ≈ 0.50 - 0.28 = 0.22", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.suptitle("Scoring a split by how much impurity it removes (illustrative example, not project data)",
             x=0.01, y=0.98, ha="left", va="top", fontweight="bold", fontsize=12.5)
save(fig, DDIR, "concept_gini_info_gain")

print("done")
