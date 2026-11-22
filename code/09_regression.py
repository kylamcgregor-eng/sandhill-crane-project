"""
09_regression.py  -  Module 4: Linear Regression and Logistic Regression.

Two independent models, each with its own labeled split:

  Linear regression   predicts log10(total_count) - a numeric target - from six weather
                       variables and day of year, using only the 6,699 site-days in
                       data/clean/site_daily_clean.csv that actually have a sighting record
                       (has_records == 1). Its own fresh 80/20 split is written to
                       reg_train.csv / reg_test.csv.

  Logistic regression  predicts is_winter (a two-class label) from lat_norm, lon_norm and
                       log10_count. It reuses the exact train/test split already written by
                       06_naive_bayes.py (nb_dt_train.csv / nb_dt_test.csv) so its result can
                       be compared, row for row, against Part 3's Naive Bayes model. Run
                       06_naive_bayes.py first.

Writes:
  images/regression/*.png
  data/clean/reg_train.csv, data/clean/reg_test.csv
  data/clean/regression_log.json
"""
from __future__ import annotations

import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

import config as C

BLUE, ORANGE, AQUA, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#7b5ea7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
RNG = C.RANDOM_SEED

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight",
})

RDIR = C.FIGS.parent / "regression"
RDIR.mkdir(parents=True, exist_ok=True)
LOG: dict = {}


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print(f"  saved {folder.name}/{name}.png")


def table_image(df: pd.DataFrame, name: str, folder, title: str, n: int = 8, max_cols: int = 9) -> None:
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


# ================================================================== LINEAR REGRESSION
# "day" here is already day-of-year (site_daily_clean.csv builds it the same way
# observations_clean.csv does, as date.dt.dayofyear), so it is reused as-is rather than
# recomputing a duplicate column under a different name.
LIN_FEATURES = ["temp_mean_c", "precip_mm", "om_snowfall_mm", "om_wind_max_kmh",
                 "om_snow_depth_max_m", "temp_7d_mean_c", "day"]


def build_linear_split():
    print("Building the linear-regression split (site-days with a sighting record)")
    df = pd.read_csv(C.CLEAN / "site_daily_clean.csv", parse_dates=["date"])
    sub = df[df["has_records"] == 1].copy()
    sub["log10_count"] = np.log10(sub["total_count"])

    keep = (["site", "date", "season", "temp_mean_c", "precip_mm", "om_snowfall_mm",
             "om_wind_max_kmh", "total_count", "log10_count"]
            + ["state", "year", "month", "day", "om_snow_depth_max_m", "temp_7d_mean_c"])
    sub = sub[keep].reset_index(drop=True)

    sample = sub.sample(n=min(400, len(sub)), random_state=RNG).sort_values("date").reset_index(drop=True)
    train, test = train_test_split(sub, test_size=0.2, random_state=RNG)

    sample.to_csv(C.CLEAN / "reg_sample.csv", index=False)
    train.to_csv(C.CLEAN / "reg_train.csv", index=False)
    test.to_csv(C.CLEAN / "reg_test.csv", index=False)

    table_image(sample, "reg_sample_preview", RDIR,
                "Sample of site-days with a sighting record (reg_sample.csv)")
    table_image(train, "reg_train_preview", RDIR, "Training set, first rows (reg_train.csv)")
    table_image(test, "reg_test_preview", RDIR, "Testing set, first rows (reg_test.csv)")

    LOG["linear_n_total"] = len(sub)
    LOG["linear_n_train"] = len(train)
    LOG["linear_n_test"] = len(test)
    LOG["linear_features"] = LIN_FEATURES
    print(f"  {len(sub):,} site-days total -> {len(train):,} train / {len(test):,} test")
    return train, test


def fit_linear(train, test):
    print("Fitting linear regression on log10(total_count)")
    Xtr, ytr = train[LIN_FEATURES].to_numpy(), train["log10_count"].to_numpy()
    Xte, yte = test[LIN_FEATURES].to_numpy(), test["log10_count"].to_numpy()

    model = LinearRegression()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae = float(mean_absolute_error(yte, pred))
    r2 = float(r2_score(yte, pred))
    print(f"  test RMSE={rmse:.3f}  MAE={mae:.3f}  R^2={r2:.3f}")

    coefs = dict(zip(LIN_FEATURES, model.coef_.tolist()))
    LOG["linear_intercept"] = float(model.intercept_)
    LOG["linear_coefficients"] = coefs
    LOG["linear_rmse"] = rmse
    LOG["linear_mae"] = mae
    LOG["linear_r2"] = r2
    for feat in ("temp_mean_c", "precip_mm"):
        pct = (10 ** coefs[feat] - 1) * 100
        print(f"  coefficient[{feat}] = {coefs[feat]:+.5f}  ->  10^coef = {10**coefs[feat]:.4f}  ({pct:+.2f}% per unit)")
        LOG.setdefault("linear_coef_pct_per_unit", {})[feat] = pct

    # -------- predicted vs. actual
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(yte, pred, s=10, color=BLUE, alpha=0.35, edgecolor="none")
    lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=ORANGE, ls="--", lw=1.4, label="perfect prediction")
    ax.set_xlabel("actual log10(total_count)"); ax.set_ylabel("predicted log10(total_count)")
    ax.set_title(f"Linear regression: predicted vs. actual (test set, R$^2$={r2:.3f})", fontsize=12)
    ax.legend(fontsize=9)
    save(fig, RDIR, "lin_pred_vs_actual")

    # -------- residuals vs. fitted (honesty check for curvature)
    resid = yte - pred
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(pred, resid, s=10, color=PURPLE, alpha=0.35, edgecolor="none")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("predicted log10(total_count)"); ax.set_ylabel("residual (actual - predicted)")
    ax.set_title("Residuals vs. fitted values", fontsize=12.5)
    save(fig, RDIR, "lin_residuals")

    # -------- coefficient bar chart
    order = np.argsort(np.abs(model.coef_))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(np.array(LIN_FEATURES)[order], model.coef_[order], color=BLUE)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("coefficient (change in log10(total_count) per unit)")
    ax.set_title("Linear regression coefficients", fontsize=12.5)
    save(fig, RDIR, "lin_coefficients")

    return model


# ================================================================== LOGISTIC REGRESSION
LOG_FEATURES = ["lat_norm", "lon_norm", "log10_count"]


def load_logistic_split():
    print("Loading the shared Naive Bayes / Decision Tree split for logistic regression")
    df_tr = pd.read_csv(C.CLEAN / "nb_dt_train.csv")
    df_te = pd.read_csv(C.CLEAN / "nb_dt_test.csv")
    df_tr["is_winter"] = (df_tr["season"] == "winter").astype(int)
    df_te["is_winter"] = (df_te["season"] == "winter").astype(int)
    sample = pd.read_csv(C.CLEAN / "nb_dt_sample.csv")
    sample["is_winter"] = (sample["season"] == "winter").astype(int)
    table_image(sample, "log_sample_preview", RDIR,
                "Sample of the labeled season data, with is_winter added (nb_dt_sample.csv)")
    table_image(df_tr, "log_train_preview", RDIR,
                "Training set, first rows (nb_dt_train.csv, shared with Part 3)")
    table_image(df_te, "log_test_preview", RDIR,
                "Testing set, first rows (nb_dt_test.csv, shared with Part 3)")
    LOG["logistic_n_train"] = len(df_tr)
    LOG["logistic_n_test"] = len(df_te)
    LOG["logistic_features"] = LOG_FEATURES
    LOG["logistic_train_winter_rate"] = float(df_tr["is_winter"].mean())
    LOG["logistic_test_winter_rate"] = float(df_te["is_winter"].mean())
    print(f"  {len(df_tr):,} train / {len(df_te):,} test  (winter share: "
          f"{df_tr['is_winter'].mean():.3f} train, {df_te['is_winter'].mean():.3f} test)")
    return df_tr, df_te


def fit_logistic(df_tr, df_te):
    print("Fitting logistic regression on is_winter")
    Xtr, ytr = df_tr[LOG_FEATURES].to_numpy(), df_tr["is_winter"].to_numpy()
    Xte, yte = df_te[LOG_FEATURES].to_numpy(), df_te["is_winter"].to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=RNG)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    baseline_acc = 1.0 - float(yte.mean())  # always predicting the majority class (not winter)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    print(f"  test accuracy={acc:.3f}  (always-not-winter baseline={baseline_acc:.3f})")
    print(f"  confusion matrix (rows=true, cols=predicted, labels=[not winter, winter]):\n{cm}")
    LOG["logistic_baseline_accuracy"] = baseline_acc

    tn, fp, fn, tp = cm.ravel()
    precision_winter = tp / (tp + fp) if (tp + fp) else 0.0
    recall_winter = tp / (tp + fn) if (tp + fn) else 0.0

    coefs = dict(zip(LOG_FEATURES, model.coef_[0].tolist()))
    odds_ratio = {k: float(np.exp(v)) for k, v in coefs.items()}
    LOG["logistic_accuracy"] = float(acc)
    LOG["logistic_confusion_matrix"] = cm.tolist()
    LOG["logistic_confusion_labels"] = ["not_winter", "winter"]
    LOG["logistic_precision_winter"] = float(precision_winter)
    LOG["logistic_recall_winter"] = float(recall_winter)
    LOG["logistic_intercept"] = float(model.intercept_[0])
    LOG["logistic_coefficients"] = coefs
    LOG["logistic_odds_ratios"] = odds_ratio
    for k, v in odds_ratio.items():
        pct = (v - 1) * 100
        print(f"  coefficient[{k}] = {coefs[k]:+.4f}  ->  odds ratio = {v:.4f}  ({pct:+.2f}% change in odds per unit)")

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not winter", "winter"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not winter", "winter"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Logistic regression confusion matrix (accuracy {acc:.1%})", fontsize=12)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > cm.max() / 2 else INK, fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="test sightings")
    save(fig, RDIR, "log_confusion_matrix")

    # -------- sigmoid illustration using the fitted log10_count coefficient (1-D slice at median lat/lon)
    med_lat, med_lon = df_tr["lat_norm"].median(), df_tr["lon_norm"].median()
    grid = np.linspace(df_tr["log10_count"].min(), df_tr["log10_count"].max(), 200)
    Xgrid = np.column_stack([np.full_like(grid, med_lat), np.full_like(grid, med_lon), grid])
    p = model.predict_proba(Xgrid)[:, 1]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(grid, p, color=BLUE, lw=2)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xlabel("log10(count)  (lat/lon held at their median)")
    ax.set_ylabel("predicted P(winter)")
    ax.set_title("Fitted sigmoid: predicted winter probability vs. flock size", fontsize=12)
    save(fig, RDIR, "log_sigmoid_slice")

    return model


def main():
    train, test = build_linear_split()
    fit_linear(train, test)
    df_tr, df_te = load_logistic_split()
    fit_logistic(df_tr, df_te)
    with open(C.CLEAN / "regression_log.json", "w") as f:
        json.dump(LOG, f, indent=2, default=str)
    print("done -> data/clean/regression_log.json")


if __name__ == "__main__":
    main()
