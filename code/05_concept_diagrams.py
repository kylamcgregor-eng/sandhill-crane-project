"""
05_concept_diagrams.py  -  Explanatory diagrams for the Clustering and PCA tabs' Overview sections.

These illustrate the methods with small synthetic examples, not the project's crane data
(the Results sections use the real data). Each panel says so in its own title.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import dendrogram, linkage

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
CDIR = C.FIGS.parent / "clustering"
PDIR = C.FIGS.parent / "pca"
CDIR.mkdir(parents=True, exist_ok=True)
PDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight",
})


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


# ------------------------------------------------------------------ 1. partitional vs. hierarchical
rng = np.random.default_rng(5612)
blobs = [rng.normal(c, 0.35, size=(14, 2)) for c in [(1, 1), (4, 1.2), (2.5, 4)]]
pts = np.vstack(blobs)
true_lab = np.repeat([0, 1, 2], 14)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
ax = axes[0]
for c, col in zip(range(3), (BLUE, ORANGE, AQUA)):
    m = true_lab == c
    ax.scatter(pts[m, 0], pts[m, 1], color=col, s=55, alpha=0.85)
    cx, cy = pts[m].mean(axis=0)
    ax.scatter([cx], [cy], color=col, marker="X", s=180, edgecolor=INK, linewidth=1)
ax.set_title("Partitional (k-means): assign every\npoint to its nearest of k centers", fontsize=11.5)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
Z = linkage(pts, method="average")
dendrogram(Z, ax=ax, no_labels=True,
           color_threshold=0.6 * max(Z[:, 2]))
ax.set_title("Hierarchical: merge the two closest\ngroups, over and over, bottom-up", fontsize=11.5)
ax.set_ylabel("distance"); ax.set_xticks([])
fig.suptitle("Two ways to group unlabeled points (illustrative example, not project data)",
             x=0.01, ha="left", fontweight="bold", fontsize=13)
save(fig, CDIR, "concept_partitional_vs_hierarchical")

# ------------------------------------------------------------------ 2. distance metrics
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
a = np.array([3.4, 1.0]); b = np.array([1.2, 2.6])
ax = axes[0]
for v, col, lab in ((a, BLUE, "point A"), (b, ORANGE, "point B")):
    ax.add_patch(FancyArrowPatch((0, 0), v, arrowstyle="-|>", mutation_scale=18, lw=2.4, color=col))
    ax.text(*(v * 1.08), lab, color=col, fontsize=10, fontweight="bold")
ax.plot([a[0], b[0]], [a[1], b[1]], ls="--", color=INK2, lw=1.6)
mid = (a + b) / 2
ax.text(*mid, f"  Euclidean distance\n  = straight-line length\n  = {np.linalg.norm(a-b):.2f}",
        color=INK2, fontsize=9.5, va="center")
ax.set_xlim(-0.3, 4); ax.set_ylim(-0.3, 3.3); ax.set_aspect("equal")
ax.set_title("Euclidean distance: how far apart\nthe two points sit", fontsize=11.5)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
for v, col, lab in ((a, BLUE, "point A"), (b, ORANGE, "point B")):
    ax.add_patch(FancyArrowPatch((0, 0), v, arrowstyle="-|>", mutation_scale=18, lw=2.4, color=col))
    ax.text(*(v * 1.08), lab, color=col, fontsize=10, fontweight="bold")
theta = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
arc = np.linspace(np.arctan2(b[1], b[0]), np.arctan2(a[1], a[0]), 30)
ax.plot(0.7 * np.cos(arc), 0.7 * np.sin(arc), color=INK2, lw=1.6)
ax.text(1.0, 0.55, f"angle θ = {theta:.0f}°\ncosine similarity\n= cos(θ) = {np.cos(np.radians(theta)):.2f}",
        color=INK2, fontsize=9.5, va="center")
ax.set_xlim(-0.3, 4); ax.set_ylim(-0.3, 3.3); ax.set_aspect("equal")
ax.set_title("Cosine similarity: how similar the two\npoints' directions are, ignoring length", fontsize=11.5)
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Two ways to compare two data points (illustrative example, not project data)",
             x=0.01, ha="left", fontweight="bold", fontsize=13)
save(fig, CDIR, "concept_distance_metrics")

# ------------------------------------------------------------------ 3. eigenvectors / eigenvalues
rng = np.random.default_rng(7)
cov = np.array([[3.0, 2.2], [2.2, 2.0]])
cloud = rng.multivariate_normal([0, 0], cov, size=400)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

fig, ax = plt.subplots(figsize=(7, 6.2))
ax.scatter(cloud[:, 0], cloud[:, 1], s=10, alpha=0.35, color=GREY, linewidths=0)
for i, (val, col, lab) in enumerate(zip(vals, (ORANGE, BLUE), ("PC1 (largest eigenvalue:\nmost variance)",
                                                                "PC2 (smaller eigenvalue:\nless variance)"))):
    v = vecs[:, i] * np.sqrt(val) * 1.8
    ax.add_patch(FancyArrowPatch((0, 0), v, arrowstyle="-|>", mutation_scale=20, lw=3, color=col))
    ax.text(*(v * 1.12), lab, color=col, fontsize=10, fontweight="bold", ha="center")
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal")
ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
ax.set_title("A cloud of points has a shape; its eigenvectors\npoint along that shape's axes (illustrative example)",
             fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
save(fig, PDIR, "concept_eigenvectors")

# ------------------------------------------------------------------ 4. dimensionality reduction
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.axis("off")
boxes = [("5 numeric\ncolumns", 0.12), ("PCA finds the\ndirections of most\nspread", 0.38),
         ("keep the first\n2-3 components", 0.64), ("a 2-D or 3-D\nplot, easy to\nsee and model", 0.88)]
y = 0.5
box_w = 0.20
for i, (label, x) in enumerate(boxes):
    ax.add_patch(plt.Rectangle((x - box_w / 2, y - 0.22), box_w, 0.44, facecolor="#cde2fb" if i % 2 == 0 else "#ffe3d1",
                                edgecolor=INK2, linewidth=1.2, transform=ax.transAxes))
    ax.text(x, y, label, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    if i < len(boxes) - 1:
        gap = box_w / 2 + 0.01
        ax.annotate("", xy=(boxes[i + 1][1] - gap, y), xytext=(x + gap, y),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
ax.set_title("Why reduce dimensions: fewer numbers that still carry most of the information",
             fontsize=12.5, loc="left")
save(fig, PDIR, "concept_dimensionality_reduction")

print("done")
