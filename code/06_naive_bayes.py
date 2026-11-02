"""
06_naive_bayes.py  -  Module 3: Naive Bayes classification.

Supervised target: season (winter / spring / summer / fall), already labeled in
observations_clean.csv via config.season_of(month). The features deliberately do NOT include
the date itself (day of year, month, week): season is defined directly from the month, so giving
a classifier the month would make the problem circular and trivial. Instead this asks a genuine
question: can the season of a sighting be predicted from where it happened and how many birds
were in it, without ever being told when it happened?

Features (all numeric and non-negative, which MultinomialNB requires):
  lat_norm, lon_norm   - location, scaled 0-1 (same columns the Clustering/PCA tab used)
  log10_count          - flock size on a log scale (count >= 1, so this is never negative)

Writes:
  data/clean/nb_dt_sample.csv, nb_dt_train.csv, nb_dt_test.csv  - labeled split shared with
                                                                   07_decision_trees.py
  images/naivebayes/*.png
  data/clean/naive_bayes_log.json
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SEASONS = ["winter", "spring", "summer", "fall"]
SEASON_COLOR = {"winter": "#5a6fb0", "spring": AQUA, "summer": ORANGE, "fall": "#a67c2e"}
FEATURES = ["lat_norm", "lon_norm", "log10_count"]
RNG = C.RANDOM_SEED

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight",
})

NDIR = C.FIGS.parent / "naivebayes"
NDIR.mkdir(parents=True, exist_ok=True)
LOG: dict = {}


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


def table_image(df: pd.DataFrame, name: str, folder, title: str, n: int = 8, max_cols: int = 9) -> None:
    """Small picture of the first rows of a dataset (never paste whole datasets into the site)."""
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


# ------------------------------------------------------------------ 1. labeled data + disjoint split
def build_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Building the labeled train/test split (shared with Decision Trees)")
    obs = pd.read_csv(C.F_OBS_CLEAN)
    df = obs[["lat", "lon", "count", "lat_norm", "lon_norm", "log10_count", "season"]].copy()

    df_tr, df_te = train_test_split(df, test_size=0.2, random_state=RNG, stratify=df["season"])
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    sample = df.groupby("season", group_keys=False)[df.columns].apply(
        lambda g: g.sample(n=min(len(g), 100), random_state=RNG)
    ).sample(frac=1, random_state=RNG).reset_index(drop=True)

    df.to_csv(C.CLEAN / "nb_dt_all_labeled.csv", index=False)
    sample.to_csv(C.CLEAN / "nb_dt_sample.csv", index=False)
    df_tr.to_csv(C.CLEAN / "nb_dt_train.csv", index=False)
    df_te.to_csv(C.CLEAN / "nb_dt_test.csv", index=False)
    print(f"  {len(df):,} labeled rows -> {len(df_tr):,} train / {len(df_te):,} test "
          f"({len(df_te) / len(df):.0%} held out), {len(sample):,}-row sample saved for preview")

    table_image(sample, "nb_dt_sample_preview", NDIR,
                "Sample of the labeled season data (nb_dt_sample.csv)")
    table_image(df_tr, "nb_dt_train_preview", NDIR,
                "Training set, first rows (nb_dt_train.csv)")
    table_image(df_te, "nb_dt_test_preview", NDIR,
                "Testing set, first rows (nb_dt_test.csv)")

    LOG["n_total"] = len(df)
    LOG["n_train"] = len(df_tr)
    LOG["n_test"] = len(df_te)
    LOG["train_season_counts"] = df_tr["season"].value_counts().to_dict()
    LOG["test_season_counts"] = df_te["season"].value_counts().to_dict()
    return df_tr, df_te


# ------------------------------------------------------------------ 2. Multinomial and Bernoulli NB
def run_naive_bayes(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> None:
    Xtr = df_tr[FEATURES].to_numpy()
    ytr = df_tr["season"].to_numpy()
    Xte = df_te[FEATURES].to_numpy()
    yte = df_te["season"].to_numpy()
    assert Xtr.min() >= 0 and Xte.min() >= 0, "MultinomialNB requires non-negative features"

    # -------- majority-class baseline, for context
    majority = pd.Series(ytr).value_counts().idxmax()
    baseline_acc = accuracy_score(yte, [majority] * len(yte))

    # -------- Multinomial NB (the required method)
    print("Multinomial Naive Bayes")
    mnb = MultinomialNB()
    mnb.fit(Xtr, ytr)
    pred_mnb = mnb.predict(Xte)
    acc_mnb = accuracy_score(yte, pred_mnb)
    cm_mnb = confusion_matrix(yte, pred_mnb, labels=SEASONS)
    report_mnb = classification_report(yte, pred_mnb, labels=SEASONS, output_dict=True, zero_division=0)
    print(f"  accuracy: {acc_mnb:.3f}  (majority-class baseline: {baseline_acc:.3f})")

    # -------- Bernoulli NB, for comparison (binarize each feature at its training median)
    print("Bernoulli Naive Bayes (features binarized at the training median)")
    medians = np.median(Xtr, axis=0)
    Xtr_bin = (Xtr > medians).astype(int)
    Xte_bin = (Xte > medians).astype(int)
    bnb = BernoulliNB()
    bnb.fit(Xtr_bin, ytr)
    pred_bnb = bnb.predict(Xte_bin)
    acc_bnb = accuracy_score(yte, pred_bnb)
    print(f"  accuracy: {acc_bnb:.3f}")

    LOG["features"] = FEATURES
    LOG["majority_class"] = majority
    LOG["baseline_accuracy"] = baseline_acc
    LOG["multinomial_accuracy"] = acc_mnb
    LOG["multinomial_confusion_matrix"] = cm_mnb.tolist()
    LOG["multinomial_confusion_labels"] = SEASONS
    LOG["multinomial_classification_report"] = report_mnb
    LOG["bernoulli_binarize_thresholds"] = dict(zip(FEATURES, medians.tolist()))
    LOG["bernoulli_accuracy"] = acc_bnb

    # -------- figure: confusion matrix heatmap (Multinomial NB)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_mnb, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(SEASONS, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(SEASONS)
    ax.set_xlabel("predicted season"); ax.set_ylabel("true season")
    ax.set_title("Multinomial NB confusion matrix (test set)", fontsize=12.5)
    for i in range(4):
        for j in range(4):
            v = cm_mnb[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > cm_mnb.max() / 2 else INK, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="test sightings")
    save(fig, NDIR, "nb_confusion_matrix")

    # -------- figure: accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    names = ["Majority-class\nbaseline", "Bernoulli NB\n(binarized)", "Multinomial NB\n(required method)"]
    accs = [baseline_acc, acc_bnb, acc_mnb]
    cols = [GREY, PURPLE, BLUE]
    bars = ax.bar(names, accs, color=cols, width=0.6)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Predicting season from location and flock size", fontsize=12.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.015, f"{a:.1%}", ha="center", fontsize=10.5, fontweight="bold")
    save(fig, NDIR, "nb_accuracy_comparison")

    # -------- figure: test-set map, correct vs. misclassified (Multinomial NB)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    correct = pred_mnb == yte
    for s in SEASONS:
        m = (yte == s) & correct
        ax.scatter(df_te.loc[m, "lon"], df_te.loc[m, "lat"], s=10, alpha=0.35, color=SEASON_COLOR[s], label=f"{s} (correct)")
    for s in SEASONS:
        m = (yte == s) & ~correct
        ax.scatter(df_te.loc[m, "lon"], df_te.loc[m, "lat"], s=22, marker="x", alpha=0.6, color=SEASON_COLOR[s])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Multinomial NB test predictions by location", fontsize=12.5)
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    save(fig, NDIR, "nb_predictions_map")


def main():
    df_tr, df_te = build_split()
    run_naive_bayes(df_tr, df_te)
    with open(C.CLEAN / "naive_bayes_log.json", "w") as f:
        json.dump(LOG, f, indent=2, default=str)
    print("done -> data/clean/naive_bayes_log.json")


if __name__ == "__main__":
    main()
