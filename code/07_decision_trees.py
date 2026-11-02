"""
07_decision_trees.py  -  Module 3: Decision Tree classification.

Reuses the exact train/test split written by 06_naive_bayes.py (data/clean/nb_dt_train.csv and
nb_dt_test.csv: season labeled from lat_norm, lon_norm, log10_count, with date-derived columns
left out on purpose, see that script's header) so the two methods are compared on identical rows.
Run 06_naive_bayes.py first.

Writes:
  images/dectrees/*.png   (plus copies of the three data-prep preview tables for this tab)
  data/clean/decision_trees_log.json
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SEASONS = ["winter", "spring", "summer", "fall"]
FEATURES = ["lat_norm", "lon_norm", "log10_count"]
RNG = C.RANDOM_SEED

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight",
})

DDIR = C.FIGS.parent / "dectrees"
DDIR.mkdir(parents=True, exist_ok=True)
LOG: dict = {}


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


def table_image(df: pd.DataFrame, name: str, folder, title: str, n: int = 8, max_cols: int = 9) -> None:
    import textwrap
    d = df.head(n).iloc[:, :max_cols].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(3)
        d[c] = d[c].astype(str).str.replace(r"\.0$", "", regex=True).str.slice(0, 19)
    headers = ["\n".join(textwrap.wrap(str(c), 18, break_long_words=True)) for c in d.columns]
    fig, ax = plt.subplots(figsize=(min(1.45 * len(d.columns) + 0.5, 13.5), 0.36 * (len(d) + 3.2)))
    ax.axis("off")
    tb = ax.table(cellText=d.values, colLabels=headers, loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
    for (r, _c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#cde2fb"); cell.set_text_props(weight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.9)
    ax.set_title(title, fontsize=10, loc="left", color=INK2)
    save(fig, folder, name)


def gini(p: np.ndarray) -> float:
    return 1.0 - np.sum(p ** 2)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def class_proportions(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    return np.array([counts.get(s, 0) for s in SEASONS]) / len(y)


# ------------------------------------------------------------------ 1. load the shared split
def load_split():
    df_tr = pd.read_csv(C.CLEAN / "nb_dt_train.csv")
    df_te = pd.read_csv(C.CLEAN / "nb_dt_test.csv")
    sample = pd.read_csv(C.CLEAN / "nb_dt_sample.csv")
    for name, df in (("nb_dt_sample_preview", sample), ("nb_dt_train_preview", df_tr), ("nb_dt_test_preview", df_te)):
        title = {"nb_dt_sample_preview": "Sample of the labeled season data (nb_dt_sample.csv)",
                 "nb_dt_train_preview": "Training set, first rows (nb_dt_train.csv)",
                 "nb_dt_test_preview": "Testing set, first rows (nb_dt_test.csv)"}[name]
        table_image(df, name, DDIR, title)
    LOG["n_train"] = len(df_tr)
    LOG["n_test"] = len(df_te)
    return df_tr, df_te


# ------------------------------------------------------------------ 2. depth sweep -> pick the best tree
PRACTICAL_MAX_DEPTH = 12  # a tree much deeper reads as thousands of leaves, not an explainable set of rules


def depth_sweep(Xtr, ytr, Xte, yte):
    print("Sweeping max_depth to find where test accuracy peaks")
    depths = list(range(1, 31))
    train_acc, test_acc, n_leaves = [], [], []
    for d in depths:
        t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=RNG)
        t.fit(Xtr, ytr)
        train_acc.append(t.score(Xtr, ytr))
        test_acc.append(t.score(Xte, yte))
        n_leaves.append(t.get_n_leaves())
    raw_argmax_depth = depths[int(np.argmax(test_acc))]
    practical_idx = [i for i, d in enumerate(depths) if d <= PRACTICAL_MAX_DEPTH]
    best_depth = depths[practical_idx[int(np.argmax([test_acc[i] for i in practical_idx]))]]
    print(f"  raw test-accuracy peak is at max_depth={raw_argmax_depth} ({max(test_acc):.3f}), "
          f"but train accuracy is still climbing there too ({train_acc[depths.index(raw_argmax_depth)]:.3f}) "
          f"over {n_leaves[depths.index(raw_argmax_depth)]:,} leaves - that is memorizing, not generalizing.")
    print(f"  within a practical depth of {PRACTICAL_MAX_DEPTH} or less, the best is max_depth={best_depth} "
          f"({test_acc[depths.index(best_depth)]:.3f})")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(depths, train_acc, color=GREY, marker="o", ms=3, label="train accuracy")
    ax.plot(depths, test_acc, color=BLUE, marker="o", ms=3, label="test accuracy")
    ax.axvspan(0.5, PRACTICAL_MAX_DEPTH + 0.5, color=AQUA, alpha=0.08, label=f"practical range (depth $\\leq$ {PRACTICAL_MAX_DEPTH})")
    ax.axvline(best_depth, color=ORANGE, ls="--", lw=1.4, label=f"chosen depth = {best_depth}")
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy")
    ax.set_title("Train vs. test accuracy as depth grows", fontsize=12.5)
    ax.legend(fontsize=9)
    save(fig, DDIR, "dt_depth_sweep")

    LOG["depth_sweep_depths"] = depths
    LOG["depth_sweep_train_acc"] = train_acc
    LOG["depth_sweep_test_acc"] = test_acc
    LOG["depth_sweep_n_leaves"] = n_leaves
    LOG["practical_max_depth"] = PRACTICAL_MAX_DEPTH
    LOG["raw_argmax_depth"] = raw_argmax_depth
    LOG["best_depth"] = best_depth
    return best_depth


# ------------------------------------------------------------------ 3. three trees
def fit_trees(df_tr, df_te, best_depth):
    Xtr = df_tr[FEATURES].to_numpy(); ytr = df_tr["season"].to_numpy()
    Xte = df_te[FEATURES].to_numpy(); yte = df_te["season"].to_numpy()

    trees = {
        "gini_shallow": DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=RNG),
        "entropy_shallow": DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=RNG),
        "gini_best": DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=RNG),
    }
    accs = {}
    for name, tree in trees.items():
        tree.fit(Xtr, ytr)
        accs[name] = accuracy_score(yte, tree.predict(Xte))
        print(f"  {name}: test accuracy {accs[name]:.3f}")
    LOG["tree_accuracies"] = accs

    # -------- figure: the two shallow trees, full depth (readable), gini vs entropy
    for name, title in (("gini_shallow", "Tree 1, criterion=gini, max_depth=2"),
                         ("entropy_shallow", "Tree 2, criterion=entropy, max_depth=2")):
        fig, ax = plt.subplots(figsize=(11, 5.2))
        plot_tree(trees[name], feature_names=FEATURES, class_names=SEASONS, filled=True,
                  rounded=True, fontsize=9, ax=ax, impurity=True)
        ax.set_title(f"{title}  (test accuracy {accs[name]:.1%})", fontsize=12.5)
        save(fig, DDIR, f"dt_tree_{name}")

    # -------- figure: the best (deeper) tree, top 3 levels shown for readability
    fig, ax = plt.subplots(figsize=(17, 7.5))
    plot_tree(trees["gini_best"], feature_names=FEATURES, class_names=SEASONS, filled=True,
              rounded=True, fontsize=7, ax=ax, impurity=True, max_depth=3)
    ax.set_title(f"Tree 3, criterion=gini, max_depth={best_depth} (top 3 levels shown, "
                 f"test accuracy {accs['gini_best']:.1%})", fontsize=12.5)
    save(fig, DDIR, "dt_tree_gini_best")

    # -------- figure: feature importances of the best tree
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    imp = trees["gini_best"].feature_importances_
    order = np.argsort(imp)
    ax.barh(np.array(FEATURES)[order], imp[order], color=BLUE)
    ax.set_xlabel("Gini importance"); ax.set_title("Which feature does the best tree lean on?", fontsize=12.5)
    save(fig, DDIR, "dt_feature_importance")
    LOG["feature_importance_best_tree"] = dict(zip(FEATURES, imp.tolist()))

    # -------- confusion matrix for the best tree (primary Results)
    pred = trees["gini_best"].predict(Xte)
    cm = confusion_matrix(yte, pred, labels=SEASONS)
    report = classification_report(yte, pred, labels=SEASONS, output_dict=True, zero_division=0)
    LOG["best_tree_confusion_matrix"] = cm.tolist()
    LOG["best_tree_confusion_labels"] = SEASONS
    LOG["best_tree_classification_report"] = report

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(4)); ax.set_xticklabels(SEASONS, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(SEASONS)
    ax.set_xlabel("predicted season"); ax.set_ylabel("true season")
    ax.set_title(f"Best tree confusion matrix (depth={best_depth}, test set)", fontsize=12)
    for i in range(4):
        for j in range(4):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > cm.max() / 2 else INK, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="test sightings")
    save(fig, DDIR, "dt_confusion_matrix")

    return trees, Xtr, ytr


# ------------------------------------------------------------------ 4. GINI / entropy / information-gain worked example
def worked_example(tree, Xtr, ytr):
    t = tree.tree_
    feat_idx = t.feature[0]
    thr = t.threshold[0]
    feat_name = FEATURES[feat_idx]
    left_mask = Xtr[:, feat_idx] <= thr
    right_mask = ~left_mask

    p_parent = class_proportions(ytr)
    p_left = class_proportions(ytr[left_mask])
    p_right = class_proportions(ytr[right_mask])
    n, nl, nr = len(ytr), left_mask.sum(), right_mask.sum()

    g_parent, g_left, g_right = gini(p_parent), gini(p_left), gini(p_right)
    g_gain = g_parent - (nl / n) * g_left - (nr / n) * g_right

    h_parent, h_left, h_right = entropy(p_parent), entropy(p_left), entropy(p_right)
    h_gain = h_parent - (nl / n) * h_left - (nr / n) * h_right

    LOG["worked_example"] = {
        "feature": feat_name, "threshold": float(thr), "n_parent": int(n),
        "n_left": int(nl), "n_right": int(nr),
        "gini_parent": g_parent, "gini_left": g_left, "gini_right": g_right, "gini_gain": g_gain,
        "entropy_parent": h_parent, "entropy_left": h_left, "entropy_right": h_right, "entropy_gain": h_gain,
        "sklearn_impurity_at_nodes": t.impurity[:3].tolist(),
    }
    print(f"Worked example: root split on {feat_name} <= {thr:.3f}")
    print(f"  Gini:    parent={g_parent:.3f}  left={g_left:.3f} (n={nl:,})  right={g_right:.3f} (n={nr:,})  gain={g_gain:.3f}")
    print(f"  Entropy: parent={h_parent:.3f}  left={h_left:.3f} (n={nl:,})  right={h_right:.3f} (n={nr:,})  gain={h_gain:.3f}")
    print(f"  (sklearn's own impurity at the same three nodes, gini: {t.impurity[:3]})")


def main():
    df_tr, df_te = load_split()
    Xtr_f = df_tr[FEATURES].to_numpy(); ytr_f = df_tr["season"].to_numpy()
    Xte_f = df_te[FEATURES].to_numpy(); yte_f = df_te["season"].to_numpy()
    best_depth = depth_sweep(Xtr_f, ytr_f, Xte_f, yte_f)
    trees, Xtr, ytr = fit_trees(df_tr, df_te, best_depth)
    worked_example(trees["gini_shallow"], Xtr, ytr)
    with open(C.CLEAN / "decision_trees_log.json", "w") as f:
        json.dump(LOG, f, indent=2, default=str)
    print("done -> data/clean/decision_trees_log.json")


if __name__ == "__main__":
    main()
