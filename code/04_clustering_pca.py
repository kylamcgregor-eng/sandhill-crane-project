"""
04_clustering_pca.py  -  Module 2: k-means + hierarchical clustering, and PCA.

Both methods run on the same unlabeled numeric description of each cleaned crane sighting:
where it was (latitude, longitude), when it was (day of year, year), and how many birds were
counted. No label column (season, phase, count_class, ...) is used anywhere in this script;
the labels in the cleaned file are only used afterward to help interpret the clusters.

Writes:
  data/clean/cluster_features.csv          - the unlabeled numeric table clustering/PCA use
  images/clustering/*.png                  - clustering figures
  images/pca/*.png                         - PCA figures
  data/clean/clustering_pca_log.json       - every number this script prints, for the site text
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
CLUSTER_COLORS = [BLUE, ORANGE, AQUA, PURPLE, "#c93a39", "#2f8f6b", "#a67c2e", "#5a6fb0"]
RNG = C.RANDOM_SEED

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight",
})

CDIR = C.FIGS.parent / "clustering"
PDIR = C.FIGS.parent / "pca"
CDIR.mkdir(parents=True, exist_ok=True)
PDIR.mkdir(parents=True, exist_ok=True)
LOG: dict = {}


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


# ------------------------------------------------------------------ 1. build the unlabeled feature table
def build_features() -> pd.DataFrame:
    obs = pd.read_csv(C.F_OBS_CLEAN, parse_dates=["date"])
    df = pd.DataFrame({
        "lat_norm": obs["lat_norm"],
        "lon_norm": obs["lon_norm"],
        "day_norm": obs["day_norm"],
        "year_norm": (obs["year"] - obs["year"].min()) / (obs["year"].max() - obs["year"].min()),
        "count_norm": (obs["log10_count"] - obs["log10_count"].min())
                      / (obs["log10_count"].max() - obs["log10_count"].min()),
    })
    assert df.isna().sum().sum() == 0
    df.to_csv(C.CLEAN / "cluster_features.csv", index=False)
    print(f"  wrote {len(df):,} rows -> data/clean/cluster_features.csv")
    # keep the season label (not used by any model) only to color plots afterward
    meta = obs[["season", "lat", "lon", "year"]].reset_index(drop=True)
    LOG["n_rows"] = int(len(df))
    return df, meta


# ------------------------------------------------------------------ 2. k-means
def run_kmeans(X: np.ndarray, df: pd.DataFrame, meta: pd.DataFrame):
    print("K-means")
    rng = np.random.default_rng(RNG)
    sample_idx = rng.choice(len(X), size=4000, replace=False)
    Xs = X[sample_idx]

    sil_scores = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit(Xs)
        sil_scores[k] = float(silhouette_score(Xs, km.labels_))
    # The score barely moves across k=2..10 (a narrow, noisy band -- see the log), so the raw
    # maximum sits at the edge of the tested range and is not a meaningful "best k". We instead
    # take the best k within an interpretable range (3 to 8, since the site tracks 7 locations).
    practical_range = {k: v for k, v in sil_scores.items() if 3 <= k <= 8}
    best_k = max(practical_range, key=practical_range.get)
    raw_best_k = max(sil_scores, key=sil_scores.get)
    LOG["kmeans_silhouette_by_k"] = sil_scores
    LOG["kmeans_best_k"] = best_k
    LOG["kmeans_raw_argmax_k"] = raw_best_k
    print(f"  silhouette by k: { {k: round(v, 3) for k, v in sil_scores.items()} }")
    print(f"  raw argmax k (edge of range, not used): {raw_best_k}; practical best k (3-8): {best_k}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ks = list(sil_scores); vs = [sil_scores[k] for k in ks]
    ax.axvspan(3, 8, color=BLUE, alpha=0.08, zorder=0, label="interpretable range used (3-8)")
    ax.plot(ks, vs, color=BLUE, marker="o", lw=2, zorder=2)
    ax.scatter([best_k], [sil_scores[best_k]], color=ORANGE, s=100, zorder=5, label=f"chosen: k={best_k}")
    ax.set_xlabel("number of clusters (k)"); ax.set_ylabel("mean silhouette score")
    ax.set_title("Silhouette score by k (k-means, 4,000-point sample)")
    ax.text(0.015, 0.03, "the score stays in a narrow, noisy band across k=2-10;\nk=10 scores highest, but k=6 is best within the practical range (3-8)",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.legend(loc="upper left", fontsize=9); save(fig, CDIR, "kmeans_silhouette_by_k")

    compare_k = [3, best_k, 8] if best_k not in (3, 8) else sorted({3, best_k, 8})
    km_fits = {}
    for k in compare_k:
        km = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit(X)
        km_fits[k] = km
        LOG.setdefault("kmeans_cluster_sizes", {})[k] = pd.Series(km.labels_).value_counts().sort_index().to_dict()

    fig, axes = plt.subplots(1, len(compare_k), figsize=(5.2 * len(compare_k), 4.6), sharex=True, sharey=True)
    for ax, k in zip(np.atleast_1d(axes), compare_k):
        lab = km_fits[k].labels_
        for c in range(k):
            m = lab == c
            ax.scatter(meta.loc[m, "lon"], meta.loc[m, "lat"], s=4, alpha=0.35,
                       color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], linewidths=0)
        ax.set_title(f"k = {k}", fontsize=12)
        ax.set_xlabel("longitude")
    np.atleast_1d(axes)[0].set_ylabel("latitude")
    fig.suptitle("K-means clusters, by sighting location (clustering also used day, year, flock size)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    save(fig, CDIR, "kmeans_three_k_map")

    # one more clustering image, not a map: how the best-k clusters differ by day of year
    best_lab = km_fits[best_k].labels_
    fig, ax = plt.subplots(figsize=(8, 4.8))
    day = df["day_norm"] * 365
    box = [day[best_lab == c] for c in range(best_k)]
    bp = ax.boxplot(box, patch_artist=True, showfliers=False, tick_labels=[f"cluster {c}" for c in range(best_k)])
    for patch, c in zip(bp["boxes"], range(best_k)):
        patch.set_facecolor(CLUSTER_COLORS[c % len(CLUSTER_COLORS)]); patch.set_alpha(0.75)
    ax.set_ylabel("day of year"); ax.set_ylim(0, 366)
    ax.set_title(f"K-means clusters (k={best_k}) by day of year (middle 50% of each cluster; outlier points not drawn)")
    save(fig, CDIR, "kmeans_best_k_day_of_year")

    return best_k, sil_scores, km_fits[best_k]


# ------------------------------------------------------------------ 3. hierarchical clustering (cosine)
def run_hclust(X: np.ndarray, df: pd.DataFrame, meta: pd.DataFrame, kmeans_best_k: int, kmeans_labels: np.ndarray):
    print("Hierarchical clustering (cosine distance)")
    rng = np.random.default_rng(RNG)
    idx = rng.choice(len(X), size=220, replace=False)
    Xh = X[idx]

    Z = linkage(Xh, method="average", metric="cosine")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]), no_labels=True)
    ax.set_xlabel("sighting (220-point sample)"); ax.set_ylabel("cosine distance")
    ax.set_title("Hierarchical clustering dendrogram (average linkage, cosine distance)")
    save(fig, CDIR, "hclust_dendrogram")

    sil_h = {}
    for k in range(2, 9):
        labels_h = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels_h)) < 2:
            continue
        sil_h[k] = float(silhouette_score(Xh, labels_h, metric="cosine"))
    best_k_h = max(sil_h, key=sil_h.get)
    LOG["hclust_silhouette_by_k"] = sil_h
    LOG["hclust_best_k"] = best_k_h
    print(f"  hclust silhouette by k: { {k: round(v, 3) for k, v in sil_h.items()} }")
    print(f"  hclust suggested k: {best_k_h}")

    hc_labels_best = fcluster(Z, t=best_k_h, criterion="maxclust")
    km_same_sample = KMeans(n_clusters=best_k_h, random_state=RNG, n_init=10).fit(Xh)
    ari = float(adjusted_rand_score(km_same_sample.labels_, hc_labels_best))
    LOG["kmeans_vs_hclust_ari_at_hclust_k"] = ari
    print(f"  agreement (Adjusted Rand Index) between k-means and h-clust at k={best_k_h}: {ari:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for c in sorted(set(hc_labels_best)):
        m = hc_labels_best == c
        ax.scatter(np.array(meta["lon"])[idx][m], np.array(meta["lat"])[idx][m], s=28,
                   color=CLUSTER_COLORS[(c - 1) % len(CLUSTER_COLORS)], label=f"cluster {c}", alpha=0.85)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Hierarchical clusters at its suggested k = {best_k_h} (same 220-point sample)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, CDIR, "hclust_best_k_map")

    return best_k_h, sil_h, ari


# ------------------------------------------------------------------ 4. PCA
def run_pca(X: np.ndarray, df: pd.DataFrame, meta: pd.DataFrame, kmeans_labels: np.ndarray, kmeans_k: int):
    print("PCA")
    pca = PCA(n_components=X.shape[1], random_state=RNG).fit(X)
    ev = pca.explained_variance_ratio_
    LOG["pca_explained_variance_ratio"] = [float(v) for v in ev]
    LOG["pca_cumulative_2pc"] = float(ev[:2].sum())
    LOG["pca_cumulative_3pc"] = float(ev[:3].sum())
    loadings = pd.DataFrame(pca.components_.T, index=df.columns,
                             columns=[f"PC{i+1}" for i in range(X.shape[1])])
    LOG["pca_loadings_pc1_pc2"] = loadings[["PC1", "PC2"]].round(3).to_dict()
    print("  explained variance ratio:", [round(v, 3) for v in ev])
    print(loadings.round(3))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    comps = np.arange(1, len(ev) + 1)
    ax.bar(comps, ev, color=BLUE)
    ax.plot(comps, np.cumsum(ev), color=ORANGE, marker="o", lw=2, label="cumulative")
    ax.set_xticks(comps); ax.set_xlabel("principal component"); ax.set_ylabel("share of variance explained")
    ax.set_title("PCA scree plot: variance explained by each component")
    ax.legend(); save(fig, PDIR, "pca_scree")

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    circle = plt.Circle((0, 0), 1, fill=False, color=GREY, lw=1)
    ax.add_artist(circle)
    # day_norm and count_norm point in nearly the same direction, so their default label positions
    # land close together; nudge them apart sideways so both stay readable.
    label_offset = {"day_norm": (-0.09, 0.0), "count_norm": (0.09, 0.03)}
    for feat in loadings.index:
        x, y = loadings.loc[feat, "PC1"], loadings.loc[feat, "PC2"]
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
        dx, dy = label_offset.get(feat, (0, 0))
        ax.text(x * 1.15 + dx, y * 1.15 + dy, feat, fontsize=10, ha="center", va="center", color=INK)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.set_title("PCA loadings: how each variable relates to PC1 and PC2")
    save(fig, PDIR, "pca_loadings_circle")

    rng = np.random.default_rng(RNG)
    idx = rng.choice(len(X), size=6000, replace=False)
    scores = pca.transform(X)[idx]
    season = np.array(meta["season"])[idx]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for s, col in zip(sorted(set(season)), [BLUE, ORANGE, AQUA]):
        m = season == s
        ax.scatter(scores[m, 0], scores[m, 1], s=5, alpha=0.3, color=col, label=s, linewidths=0)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.set_title("Sightings projected onto the first two principal components")
    leg = ax.legend(fontsize=8, markerscale=3)
    save(fig, PDIR, "pca_scores_by_season")

    return pca, loadings


# ------------------------------------------------------------------ 5. SVD, confirmed against the PCA above
def run_svd_check(X: np.ndarray, pca: PCA, loadings: pd.DataFrame):
    """PCA (above) is computed by scikit-learn using SVD internally. This reruns the decomposition
    explicitly with numpy's SVD on the same standardized data and checks that it reproduces the
    same explained variance and the same components (up to sign, since a singular/eigenvector and
    its negation describe the same axis)."""
    print("SVD (confirming it reproduces the PCA results above)")
    n = X.shape[0]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    svd_explained_variance = (S ** 2) / (n - 1)
    svd_ratio = svd_explained_variance / svd_explained_variance.sum()
    pca_ratio = pca.explained_variance_ratio_
    max_ratio_diff = float(np.max(np.abs(svd_ratio - pca_ratio)))

    # SVD's sign for each component is arbitrary (V and -V describe the same axis); align each
    # row of Vt to the matching PCA component before comparing them directly.
    Vt_aligned = Vt.copy()
    for i in range(Vt.shape[0]):
        if np.dot(Vt[i], pca.components_[i]) < 0:
            Vt_aligned[i] *= -1
    max_loading_diff = float(np.max(np.abs(Vt_aligned - pca.components_)))

    LOG["svd_singular_values"] = [float(v) for v in S]
    LOG["svd_explained_variance_ratio"] = [float(v) for v in svd_ratio]
    LOG["svd_vs_pca_max_variance_ratio_diff"] = max_ratio_diff
    LOG["svd_vs_pca_max_loading_diff"] = max_loading_diff

    print(f"  singular values: {[round(v, 1) for v in S]}")
    print(f"  SVD-derived explained variance ratio: {[round(v, 3) for v in svd_ratio]}")
    print(f"  max difference vs. scikit-learn PCA's explained variance ratio: {max_ratio_diff:.2e}")
    print(f"  max difference vs. scikit-learn PCA's loadings (after sign alignment): {max_loading_diff:.2e}")


def preview_table(df: pd.DataFrame):
    d = df.sample(8, random_state=RNG).sort_index().round(3)
    fig, ax = plt.subplots(figsize=(min(1.5 * len(d.columns) + 0.5, 11), 0.38 * (len(d) + 3)))
    ax.axis("off")
    tb = ax.table(cellText=d.values, colLabels=list(d.columns), loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.4)
    for (r, _c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#cde2fb"); cell.set_text_props(weight="bold", color=INK)
    ax.set_title("cluster_features.csv (first rows) - unlabeled numeric data used by clustering and PCA",
                 fontsize=10, loc="left", color=INK2)
    save(fig, CDIR, "cluster_features_preview")


def main():
    df, meta = build_features()
    preview_table(df)
    X = StandardScaler().fit_transform(df.values)

    best_k, sil, km_fit = run_kmeans(X, df, meta)
    best_k_h, sil_h, ari = run_hclust(X, df, meta, best_k, km_fit.labels_)
    pca, loadings = run_pca(X, df, meta, km_fit.labels_, best_k)
    run_svd_check(X, pca, loadings)

    (C.CLEAN / "clustering_pca_log.json").write_text(json.dumps(LOG, indent=2))
    print("\nLog saved to data/clean/clustering_pca_log.json")


if __name__ == "__main__":
    main()
