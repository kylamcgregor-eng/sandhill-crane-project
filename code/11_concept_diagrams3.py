"""
11_concept_diagrams3.py  -  Explanatory diagrams for the Regression and AdaBoost Overview
sections (Module 4). Illustrate the methods with small synthetic examples, not the project's crane
data (the Results sections on each tab use the real data, built by 09_regression.py and
10_adaboost.py). Each panel says so in its own title.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
RDIR = C.FIGS.parent / "regression"
ADIR = C.FIGS.parent / "adaboost"
RDIR.mkdir(parents=True, exist_ok=True)
ADIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight",
})


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


# ------------------------------------------------------------------ 1. Linear regression: least squares fit
rng = np.random.default_rng(5612)
x = np.linspace(0, 10, 16)
y = 2.2 * x + 3 + rng.normal(0, 3.2, size=x.size)
b1, b0 = np.polyfit(x, y, 1)
yhat = b0 + b1 * x

fig, ax = plt.subplots(figsize=(7.4, 4.8))
for xi, yi, yhi in zip(x, y, yhat):
    ax.plot([xi, xi], [yi, yhi], color=GREY, lw=1.2, zorder=1)
ax.scatter(x, y, color=BLUE, s=34, zorder=3, label="observed point")
ax.plot(x, yhat, color=ORANGE, lw=2.2, zorder=2, label=f"fitted line  $\\hat y = {b0:.1f} + {b1:.1f}x$")
ax.set_xlabel("predictor x"); ax.set_ylabel("target y")
ax.set_title("Linear regression: the line that minimizes the sum of squared residuals\n(illustrative example, not project data)",
             fontsize=11.5, loc="left")
ax.legend(fontsize=9, loc="upper left")
save(fig, RDIR, "concept_linreg_fit")

# ------------------------------------------------------------------ 2. Logistic regression: sigmoid + decision boundary
z = np.linspace(-8, 8, 400)
p = 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.plot(z, p, color=BLUE, lw=2.4)
ax.axhline(0.5, color=GREY, ls="--", lw=1.2)
ax.axvline(0, color=ORANGE, ls="--", lw=1.4, label="decision boundary: z = 0  (p = 0.5)")
ax.fill_between(z, 0, p, where=(z >= 0), color=ORANGE, alpha=0.08)
ax.fill_between(z, 0, p, where=(z < 0), color=BLUE, alpha=0.08)
ax.text(4.2, 0.18, "predict class 1\n(p > 0.5)", ha="center", fontsize=9.5, color=INK2)
ax.text(-4.2, 0.82, "predict class 0\n(p < 0.5)", ha="center", fontsize=9.5, color=INK2)
ax.set_xlabel("linear score  z = b0 + b1x1 + ... + bkxk"); ax.set_ylabel("sigmoid probability p = 1 / (1 + e$^{-z}$)")
ax.set_title("Logistic regression: the sigmoid turns an unbounded linear score into a 0-1 probability\n(illustrative example, not project data)",
             fontsize=11.5, loc="left")
ax.legend(fontsize=9, loc="center left")
save(fig, RDIR, "concept_sigmoid")

# ------------------------------------------------------------------ 3. Bagging vs. boosting
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.add_patch(plt.Rectangle((0.3, 2.3), 2.0, 1.4, facecolor="#cde2fb", edgecolor=INK2))
ax.text(1.3, 3.0, "training\nrows", ha="center", va="center", fontsize=9.5)
for i, yc in enumerate((5.0, 3.0, 1.0)):
    ax.annotate("", xy=(4.3, yc), xytext=(2.3, 3.0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
    ax.add_patch(plt.Rectangle((4.3, yc - 0.6), 1.9, 1.2, facecolor="#ffe3d1", edgecolor=INK2))
    ax.text(5.25, yc, f"deep tree {i+1}\n(bootstrap sample)", ha="center", va="center", fontsize=8.8)
    ax.annotate("", xy=(8.3, 3.0), xytext=(6.2, yc), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
ax.add_patch(plt.Circle((9.0, 3.0), 0.7, facecolor=AQUA, edgecolor=INK2))
ax.text(9.0, 3.0, "majority\nvote", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
ax.set_title("Bagging: independent deep trees fit in parallel\non resampled rows, then averaged", fontsize=11)

ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 11); ax.set_ylim(0, 6)
ax.add_patch(plt.Rectangle((0.2, 2.3), 1.6, 1.4, facecolor="#cde2fb", edgecolor=INK2))
ax.text(1.0, 3.0, "training\nrows", ha="center", va="center", fontsize=9)
xpos = [2.7, 4.6, 6.5]
for i, xc in enumerate(xpos):
    ax.add_patch(plt.Rectangle((xc, 2.5), 1.4, 1.0, facecolor="#ffe3d1", edgecolor=INK2))
    ax.text(xc + 0.7, 3.0, f"stump {i+1}", ha="center", va="center", fontsize=9)
    if i > 0:
        ax.annotate("reweight\nmisses", xy=(xc, 3.0), xytext=(xpos[i-1] + 1.4, 1.4),
                     ha="center", fontsize=7.6, color=INK2,
                     arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))
ax.annotate("", xy=(9.1, 3.0), xytext=(xpos[-1] + 1.4, 3.0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
ax.add_patch(plt.Circle((9.8, 3.0), 0.7, facecolor=PURPLE, edgecolor=INK2))
ax.text(9.8, 3.0, "weighted\nvote", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_title("Boosting (AdaBoost): weak stumps fit one after another,\neach reweighting the rows the last one missed", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.suptitle("Bagging trains independently to cut variance; boosting trains sequentially to cut bias\n(illustrative example, not project data)",
             x=0.01, y=0.98, ha="left", va="top", fontweight="bold", fontsize=12.5)
save(fig, ADIR, "concept_bagging_vs_boosting")

# ------------------------------------------------------------------ 4. Reweighting misclassified rows across rounds
rng = np.random.default_rng(3)
cls_a = rng.normal((3.0, 2.0), 1.3, size=(14, 2))
cls_b = rng.normal((4.6, 5.0), 1.3, size=(14, 2))
pts = np.vstack([cls_a, cls_b])
lab = np.array([0] * 14 + [1] * 14)
thr = 3.8  # a deliberately imperfect stump: split on x only, classes overlap heavily in x
pred = (pts[:, 0] > thr).astype(int)
miss = pred != lab

w1 = np.full(len(pts), 60.0)
w2 = np.where(miss, 220.0, 35.0)  # misclassified rows get a much larger weight / marker size next round

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, w, title in zip(axes, (w1, w2),
                         ("Round 1: every row starts with equal weight",
                          "Round 2: rows the stump missed are reweighted\nlarger - the next stump is fit to favor them")):
    for c, col in ((0, BLUE), (1, ORANGE)):
        m = lab == c
        ax.scatter(pts[m, 0], pts[m, 1], s=w[m], color=col, alpha=0.75, edgecolor=INK2, linewidth=0.6)
    ax.axvline(thr, color=INK, lw=1.8, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.8)
axes[0].text(thr + 0.15, 0.3, "stump's cut", fontsize=8.5, color=INK2)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.suptitle("A row's marker size is its weight - a wrong prediction grows the next\nround's weight, so later stumps focus where earlier ones failed (illustrative example)",
             x=0.01, y=0.98, ha="left", va="top", fontweight="bold", fontsize=12)
save(fig, ADIR, "concept_reweighting")

print("done")
