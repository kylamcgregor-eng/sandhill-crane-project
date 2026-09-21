"""
03_eda_figures.py  -  Exploratory data analysis figures + raw/clean data previews.

Reads data/raw and data/clean, writes PNGs to images/eda/. Every figure is wrapped so that
a missing input (for example, no eBird API key) skips that figure instead of stopping the run.

Figure colour rules: one hue for magnitude (blue ramp), blue/orange/aqua for at most three
categories, blue-red diverging with a grey midpoint for correlations. Text uses neutral ink.
"""
from __future__ import annotations

import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import config as C

# ------------------------------------------------------------------ style
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#256abf", "#9ec5f4", "#f0efec", "#f2a58e", "#c93a39"])
PHASE_COL = {"spring migration (Feb-Apr)": BLUE, "fall migration (Sep-Nov)": ORANGE,
             "breeding / winter (May-Aug, Dec-Jan)": AQUA}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 14, "axes.titlelocation": "left", "axes.labelsize": 11,
    "legend.frameon": False,
})
thousands = FuncFormatter(lambda v, _: f"{int(v):,}")
made: list[str] = []


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(C.FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    made.append(name)
    print(f"  saved {name}.png")


def figure(name):
    """Decorator: run a figure function, skip cleanly if its inputs are missing."""
    def wrap(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
            except FileNotFoundError as e:
                print(f"  - skipped {name}: input missing ({e.filename})")
            except Exception:
                print(f"  ! {name} failed:")
                traceback.print_exc()
        run.__name__ = name
        return run
    return wrap


def rounded_bars(ax, x, h, color, width=0.7):
    return ax.bar(x, h, width=width, color=color, linewidth=0)


# ------------------------------------------------------------------ data
def load():
    raw = pd.read_csv(C.F_GBIF_RAW, low_memory=False)
    obs = pd.read_csv(C.F_OBS_CLEAN, parse_dates=["date"])
    sd = pd.read_csv(C.F_SITE_DAILY_CLEAN, parse_dates=["date"])
    # true GBIF record totals per (country, year, month) - recorded during gathering
    cells = (raw.drop_duplicates(["countryCode", "year", "month"])
                [["countryCode", "year", "month", "cell_total_records"]])
    return raw, obs, sd, cells


# ------------------------------------------------------------------ figures
@figure("fig01_records_per_year")
def fig01(cells):
    d = cells.groupby("year")["cell_total_records"].sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    rounded_bars(ax, d.index, d.values, BLUE)
    ax.set_title("eBird Sandhill Crane records per year (US + Canada)")
    ax.set_xlabel("Year"); ax.set_ylabel("Records in GBIF")
    ax.yaxis.set_major_formatter(thousands); ax.set_xticks(d.index[::2])
    save(fig, "fig01_records_per_year")


@figure("fig02_records_per_month")
def fig02(cells):
    d = cells.groupby("month")["cell_total_records"].sum().reindex(range(1, 13), fill_value=0)
    cols = [BLUE if m in (2, 3, 4, 9, 10, 11) else GREY for m in d.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(MONTHS, d.values, color=cols, linewidth=0, width=0.7)
    ax.set_title("Sandhill Crane records by month, all years pooled")
    ax.set_xlabel("Month"); ax.set_ylabel("Records in GBIF"); ax.yaxis.set_major_formatter(thousands)
    ax.set_ylim(0, d.max() * 1.18)
    ax.text(0.99, 0.98, "Blue = migration months (Feb-Apr, Sep-Nov)", transform=ax.transAxes,
            ha="right", va="top", color=INK2, fontsize=10)
    save(fig, "fig02_records_per_month")


@figure("fig03_heatmap_year_month")
def fig03(cells):
    piv = (cells.groupby(["year", "month"])["cell_total_records"].sum().unstack("month")
                .reindex(columns=range(1, 13)).fillna(0))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(np.log10(piv.values + 1), aspect="auto", cmap=SEQ)
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ax.grid(False)
    ax.set_title("Records by year and month (log scale)")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("log10(records + 1)")
    cb.outline.set_visible(False)
    save(fig, "fig03_heatmap_year_month")


@figure("fig04_location_scatter")
def fig04(obs):
    d = obs.sample(min(len(obs), 40000), random_state=C.RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for phase, col in PHASE_COL.items():
        s = d[d["phase"] == phase]
        ax.scatter(s["lon"], s["lat"], s=5, alpha=0.35, color=col, label=phase, linewidths=0)
    ax.set_title("Where Sandhill Cranes were reported, by part of the year")
    ax.set_xlabel("Longitude (degrees)"); ax.set_ylabel("Latitude (degrees)")
    leg = ax.legend(markerscale=4, loc="lower left")
    for h in leg.legend_handles: h.set_alpha(1)
    save(fig, "fig04_location_scatter")


@figure("fig05_count_histogram")
def fig05(obs):
    p99 = obs["count"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(obs["log10_count"], bins=40, color=BLUE, linewidth=0)
    ax.axvline(np.log10(p99), color=ORANGE, lw=2)
    ax.text(np.log10(p99), ax.get_ylim()[1] * 0.92, f"  99th percentile = {p99:,.0f} birds",
            color=INK2, ha="left", va="top", fontsize=10)
    ticks = [0, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{10**t:,}" for t in ticks])
    ax.yaxis.set_major_formatter(thousands)
    ax.set_title("Cranes per report is heavily skewed")
    ax.set_xlabel("Cranes counted in one report (log scale)"); ax.set_ylabel("Number of reports")
    save(fig, "fig05_count_histogram")


@figure("fig06_count_by_month_box")
def fig06(obs):
    data = [obs.loc[obs["month"] == m, "log10_count"].values for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=MONTHS, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=GREY),
                    capprops=dict(color=GREY))
    for b in bp["boxes"]:
        b.set(facecolor="#9ec5f4", edgecolor=BLUE, lw=1)
    ticks = [0, 1, 2, 3, 4]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{10**t:,}" for t in ticks])
    ax.set_title("Flock size changes with the season")
    ax.set_xlabel("Month"); ax.set_ylabel("Cranes per report (log scale)")
    save(fig, "fig06_count_by_month_box")


@figure("fig07_latitude_vs_doy")
def fig07(obs):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    hb = ax.hexbin(obs["day"], obs["lat"], gridsize=(55, 28), cmap=SEQ, bins="log", mincnt=1, linewidths=0)
    ax.grid(False)
    ax.set_title("The migration wave: latitude of reports through the year")
    ax.set_xlabel("Day of year"); ax.set_ylabel("Latitude (degrees north)")
    cb = fig.colorbar(hb, ax=ax, pad=0.02); cb.set_label("Reports (log scale)"); cb.outline.set_visible(False)
    save(fig, "fig07_latitude_vs_doy")


@figure("fig08_latitude_by_week_era")
def fig08(obs):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for era, col in (("2010-2017", BLUE), ("2018-2024", ORANGE)):
        s = obs[obs["era"] == era].groupby("week")["lat"].agg(["median", "size"])
        s = s[s["size"] >= 30]
        if s.empty:
            continue
        ax.plot(s.index, s["median"], color=col, lw=2, label=era)
        ax.text(s.index[-1] + 0.4, s["median"].iloc[-1], era, color=INK2, va="center", fontsize=10)
    ax.set_xlim(1, 56)
    ax.set_title("Median latitude of reports, week by week")
    ax.set_xlabel("Week of year"); ax.set_ylabel("Median latitude (degrees north)")
    ax.legend(loc="upper right")
    save(fig, "fig08_latitude_by_week_era")


@figure("fig09_top_states")
def fig09(obs):
    d = obs["stateProvince"].value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d.index, d.values, color=BLUE, height=0.7, linewidth=0)
    ax.xaxis.set_major_formatter(thousands); ax.grid(axis="y", visible=False)
    ax.set_title("States and provinces with the most crane reports")
    ax.set_xlabel("Cleaned records in the sample"); ax.set_ylabel("")
    save(fig, "fig09_top_states")


@figure("fig10_temperature_by_site")
def fig10(sd):
    sites = list(sd["site"].unique())
    ncol = 4; nrow = int(np.ceil(len(sites) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.1 * nrow), sharex=True, sharey=True)
    for ax, site in zip(axes.flat, sites):
        g = sd[sd["site"] == site].groupby("day")["temp_mean_c"]
        med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
        ax.fill_between(med.index, lo, hi, color="#cde2fb", linewidth=0)
        ax.plot(med.index, med.values, color=BLUE, lw=1.6)
        ax.set_title(site, fontsize=10, loc="left")
    for ax in list(axes.flat)[len(sites):]:
        ax.set_visible(False)
    fig.supxlabel("Day of year", color=INK2); fig.supylabel("Daily mean temperature (°C)", color=INK2)
    fig.suptitle("Typical yearly temperature cycle at each reference site (median, 10th-90th percentile band)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    save(fig, "fig10_temperature_by_site")


@figure("fig11_correlation_heatmap")
def fig11(sd):
    spec = [("temp_mean_c", "Temp (mean)"), ("temp_7d_mean_c", "Temp (7-day)"), ("precip_mm", "Precip"),
            ("snow_mm", "Snowfall"), ("snow_depth_mm", "Snow depth"), ("wind_avg_ms", "Wind (station)"),
            ("om_snowfall_mm", "Snowfall (model)"), ("om_snow_depth_max_m", "Snow depth (model)"),
            ("om_wind_max_kmh", "Wind max (model)"), ("om_gust_max_kmh", "Gust max (model)"),
            ("day", "Day of year"),
            ("n_records", "Reports"), ("total_count", "Cranes")]
    spec = [(c, l) for c, l in spec if c in sd.columns]          # optional weather columns may be absent
    cols, labels = [c for c, _ in spec], [l for _, l in spec]
    cm = sd[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm.values, cmap=DIV, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    ax.set_title("Rank correlations: weather, season, crane reports", fontsize=13)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("Spearman correlation"); cb.outline.set_visible(False)
    save(fig, "fig11_correlation_heatmap")


def weighted_median_doy(g: pd.DataFrame) -> float:
    g = g.sort_values("day"); cum = g["total_count"].cumsum()
    return float(g.loc[cum >= cum.iloc[-1] / 2, "day"].iloc[0])


@figure("fig12_timing_trend")
def fig12(sd):
    panels = [("Platte River - Kearney", "Central Platte (Nebraska)"), ("Monte Vista NWR", "San Luis Valley (Colorado)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, (site, title) in zip(axes, panels):
        d = sd[(sd["site"] == site) & sd["month"].between(2, 4) & (sd["total_count"] > 0)]
        yrs = d.groupby("year").filter(lambda g: g["n_records"].sum() >= 10)
        med = yrs.groupby("year").apply(weighted_median_doy, include_groups=False)
        if len(med) < 3:
            ax.axis("off"); ax.set_title(title, fontsize=12)
            ax.text(0.5, 0.5, "Not enough records", ha="center", transform=ax.transAxes, color=INK2); continue
        ax.scatter(med.index, med.values, color=BLUE, s=45, zorder=3)
        slope, icpt = np.polyfit(med.index, med.values, 1)
        ax.plot(med.index, slope * med.index + icpt, color=ORANGE, lw=2, ls="--")
        ax.text(0.03, 0.05, f"trend: {slope:+.2f} days per year", transform=ax.transAxes, color=INK2, fontsize=10)
        ax.set_title(title, fontsize=12); ax.set_xlabel("Year")
    axes[0].set_ylabel("Day of year by which half of the\nFeb-Apr cranes had been reported")
    fig.suptitle("Timing of the spring peak, year by year", x=0.01, ha="left", fontweight="bold", fontsize=14)
    save(fig, "fig12_timing_trend")


@figure("fig13_spring_temperature_box")
def fig13(sd):
    d = sd[sd["month"].between(2, 4)]
    order = d.groupby("site")["temp_mean_c"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([d.loc[d["site"] == s, "temp_mean_c"] for s in order], tick_labels=order,
                    vert=False, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                    whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    for b in bp["boxes"]:
        b.set(facecolor="#9ec5f4", edgecolor=BLUE)
    ax.grid(axis="y", visible=False)
    ax.set_title("February-April daily temperature at each site")
    ax.set_xlabel("Daily mean temperature (°C)")
    save(fig, "fig13_spring_temperature_box")


@figure("fig14_presence_only_share")
def fig14(raw):
    d = raw.assign(missing=raw["individualCount"].isna()).groupby("year")["missing"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d.index, d.values, color=BLUE, lw=2, marker="o", ms=5)
    ax.set_ylim(0, max(10, d.max() * 1.2))
    ax.set_title("Reports with no crane count (presence only)")
    ax.set_xlabel("Year"); ax.set_ylabel("Share of raw records (%)")
    save(fig, "fig14_presence_only_share")


@figure("fig15_counts_vs_temperature")
def fig15(sd):
    d = sd[(sd["site"] == "Platte River - Kearney") & sd["month"].between(2, 4) & (sd["max_count"] > 0)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(d["temp_7d_mean_c"], d["max_count"], s=14, alpha=0.45, color=BLUE, linewidths=0)
    ax.set_yscale("log")
    ax.set_title("Cranes reported near Kearney vs. recent temperature (Feb-Apr)")
    ax.set_xlabel("7-day mean temperature (°C)"); ax.set_ylabel("Largest single report that day (log scale)")
    save(fig, "fig15_counts_vs_temperature")


@figure("fig16_ebird_api_recent_by_state")
def fig16():
    d = pd.read_csv(C.F_EBIRD_API_CLEAN)
    g = d.groupby("state")["howMany"].agg(["size"]).sort_values("size")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(g.index, g["size"], color=BLUE, height=0.65, linewidth=0); ax.grid(axis="y", visible=False)
    ax.set_title("Live eBird API pull: recent crane reports by state")
    ax.set_xlabel("Reports in the last 30 days at time of pull")
    save(fig, "fig16_ebird_api_recent_by_state")


# ------------------------------------------------------------------ data previews (small images)
def table_image(df: pd.DataFrame, name: str, title: str, n: int = 6, max_cols: int = 9) -> None:
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
    save(fig, name)


# ------------------------------------------------------------------ land use and cranes
SHORT = {"Platte River - Kearney": "Kearney (NE)", "Platte River - Rowe Sanctuary": "Rowe Sanctuary (NE)",
         "Monte Vista NWR": "Monte Vista (CO)", "Bosque del Apache NWR": "Bosque del Apache (NM)",
         "Grays Lake NWR": "Grays Lake (ID)", "Quivira NWR": "Quivira (KS)", "Muleshoe NWR": "Muleshoe (TX)"}


@figure("fig17_landcover_change_by_site")
def fig17():
    s = pd.read_csv(C.F_SITE_LANDCOVER_CLEAN)
    a, b = s[s["year"] == 2011].set_index("site"), s[s["year"] == 2021].set_index("site")
    d = (b - a).drop(columns="year")
    order = d["pct_cropland"].sort_values().index
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(order)); h = 0.26
    for k, (col, lab, colr) in enumerate((("pct_cropland", "Cropland", BLUE), ("pct_developed", "Developed", ORANGE),
                                          ("pct_wetland", "Wetland", AQUA))):
        ax.barh(y + (k - 1) * h, d.loc[order, col], height=h, color=colr, label=lab, linewidth=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[o] for o in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in share of land within 60 km, 2011 to 2021 (percentage points)")
    ax.set_title("Land cover changed very little at the site scale")
    ax.legend(loc="lower right")
    save(fig, "fig17_landcover_change_by_site")


@figure("fig18_crane_share_of_ebird_by_site")
def fig18():
    e = pd.read_csv(C.F_EFFORT_CLEAN)
    sites = list(SHORT)
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6), sharex=True)
    for ax, site in zip(axes.flat, sites):
        g = e[e["site"] == site]
        ax.plot(g["year"], g["crane_per_1000"], color=BLUE, lw=1.8, marker="o", ms=3.5)
        ax.set_title(SHORT[site], fontsize=10.5, loc="left"); ax.set_ylim(bottom=0)
        ax.set_xticks([2010, 2015, 2020, 2024]); ax.tick_params(labelsize=9)
    for ax in list(axes.flat)[len(sites):]:
        ax.set_visible(False)
    fig.supylabel("Crane records per 1,000 eBird records (all species)", color=INK2, fontsize=11)
    fig.suptitle("Cranes as a share of all eBird reporting near each site", x=0.01, ha="left", fontweight="bold", fontsize=14)
    save(fig, "fig18_crane_share_of_ebird_by_site")


@figure("fig19_landcover_change_vs_crane_shift")
def fig19():
    c = pd.read_csv(C.F_CELL_CHANGE_CLEAN)
    c = c[c["enough_reports"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    for ax, (col, lab) in zip(axes, (("d_developed_pp", "Change in developed land in the cell (percentage points)"),
                                     ("d_wetland_pp", "Change in wetland in the cell (percentage points)"))):
        ax.scatter(c[col], c["d_share_pp"], s=22, alpha=0.5, color=BLUE, linewidths=0)
        ax.axhline(0, color=INK2, lw=0.8)
        rho = c[[col, "d_share_pp"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"rank correlation = {rho:+.2f}  (n = {len(c)} cells)", fontsize=10, color=INK2, fontweight="normal")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("Change in the cell's share of the site's crane reports\n(2010-14 to 2020-24, percentage points)")
    fig.suptitle("Did cranes shift toward or away from cells where land use changed?", x=0.01, ha="left",
                 fontweight="bold", fontsize=14)
    save(fig, "fig19_landcover_change_vs_crane_shift")


@figure("fig20_water_wetland_and_crane_presence")
def fig20():
    c = pd.read_csv(C.F_CELL_CHANGE_CLEAN)
    c["wet"] = c["pct_wetland_2021"] + c["pct_open_water_2021"]
    c["cls"] = pd.cut(c["wet"], [-0.01, 1, 5, 15, 101], labels=["under 1%", "1-5%", "5-15%", "over 15%"])
    c["any_late"] = (c["reports_late"] > 0).astype(int)
    c["has_hotspot"] = c["hotspots"] > 0
    t = c.groupby(["cls", "has_hotspot"], observed=True)["any_late"].mean().unstack() * 100
    n = c.groupby(["cls", "has_hotspot"], observed=True).size().unstack()
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    x = np.arange(len(t)); w = 0.38
    for k, (flag, lab, colr) in enumerate(((False, "no eBird hotspot in the cell", GREY), (True, "at least one eBird hotspot", BLUE))):
        bars = ax.bar(x + (k - 0.5) * w, t[flag], width=w, color=colr, label=lab, linewidth=0)
        for xi, v, nn in zip(x + (k - 0.5) * w, t[flag], n[flag]):
            ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(t.index)
    ax.set_ylim(0, 108); ax.grid(axis="x", visible=False)
    ax.set_xlabel("Share of the 5 km cell that is open water or wetland (2021)")
    ax.set_ylabel("Cells with at least one crane report,\n2020-2024 (%)")
    ax.set_title("Cranes were reported far more often where water and wetland are present")
    ax.legend(loc="upper left")
    save(fig, "fig20_water_wetland_and_crane_presence")



@figure("fig21_snow_and_wind_trends")
def fig21(sd):
    if "om_snowfall_mm" not in sd.columns:
        raise FileNotFoundError("Open-Meteo columns")
    panels = [("Nov-Mar snowfall per winter", "Winter snowfall vs. each site's 2010-2024 average (cm)", [11, 12, 1, 2, 3],
               "om_snowfall_mm", 151 / 10),
              ("Feb-Apr wind", "Spring daily maximum wind vs. each site's average (km/h)", [2, 3, 4],
               "om_wind_max_kmh", 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, (title, ylab, months, col, scale) in zip(axes, panels):
        d = sd[sd["month"].isin(months)].copy()
        d["season_year"] = d["year"] + (d["month"] >= 11).astype(int) if 11 in months else d["year"]
        g = d.groupby(["site", "season_year"])[col].mean().mul(scale).reset_index()
        g = g[g["season_year"].between(2011, 2024) if 11 in months else g["season_year"].between(2010, 2024)]
        g["anom"] = g[col] - g.groupby("site")[col].transform("mean")
        for _, gs in g.groupby("site"):
            ax.plot(gs["season_year"], gs["anom"], color=GREY, lw=0.9, alpha=0.8)
        m = g.groupby("season_year")["anom"].mean()
        ax.plot(m.index, m.values, color=BLUE, lw=2.2, marker="o", ms=4, label="average of the 7 sites")
        slope, icpt = np.polyfit(m.index, m.values, 1)
        ax.plot(m.index, slope * m.index + icpt, color=ORANGE, lw=2, ls="--")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.text(0.03, 0.05, f"trend of the average: {slope:+.2f} per year", transform=ax.transAxes, color=INK2, fontsize=10)
        ax.set_title(title, fontsize=12, loc="left"); ax.set_xlabel("Year (winter shown by the year it ends)" if 11 in months else "Year")
        ax.set_ylabel(ylab, fontsize=9.5)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Has snowfall or wind changed at the reference sites since 2010? (modelled weather)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    save(fig, "fig21_snow_and_wind_trends")


@figure("fig22_crane_reports_by_snow_and_wind")
def fig22(sd):
    if "om_snowfall_mm" not in sd.columns:
        raise FileNotFoundError("Open-Meteo columns")
    d = sd[sd["month"].isin([2, 3, 4])].copy()
    d["snowfall"] = pd.cut(d["om_snowfall_mm"], [-1, 0.05, 20, 1e9], labels=["none", "under 2 cm", "2 cm or more"])
    d["wind"] = d.groupby("site")["om_wind_max_kmh"].transform(
        lambda x: pd.qcut(x, 3, labels=["calmest third", "middle third", "windiest third"]))
    facets = [("snowfall", "Snowfall that day"), ("wind", "Wind that day (within each site)")]
    if "om_snow_depth_max_m" in d.columns:
        d["depth"] = pd.cut(d["om_snow_depth_max_m"], [-1, 0.005, 0.05, 10], labels=["none", "under 5 cm", "5 cm or more"])
        facets.insert(1, ("depth", "Snow on the ground (max that day)"))
    fig, axes = plt.subplots(1, len(facets), figsize=(4.6 * len(facets), 5), sharey=True)
    for ax, (col, lab) in zip(np.atleast_1d(axes), facets):
        base = d.groupby("site")["has_records"].mean()
        rate = d.groupby(["site", col], observed=True)["has_records"].mean().unstack()
        ratio = rate.div(base, axis=0)
        x = np.arange(ratio.shape[1])
        ax.bar(x, ratio.mean(), color=BLUE, width=0.6, linewidth=0)
        for i, c in enumerate(ratio.columns):
            ax.scatter(np.full(len(ratio), i) + np.random.default_rng(5612).uniform(-0.12, 0.12, len(ratio)),
                       ratio[c], color=ORANGE, s=22, zorder=3, label="one site" if i == 0 else None)
        ax.axhline(1, color=INK2, lw=0.9, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(ratio.columns, fontsize=9); ax.set_xlabel(lab)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Days with a crane report, relative to a site's\naverage Feb-Apr day (1.0 = average)")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Crane reports on calm versus stormy spring days (Feb-Apr, 2010-2024)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    save(fig, "fig22_crane_reports_by_snow_and_wind")


def previews(raw, obs, sd):
    print("Data preview images")
    table_image(raw[["gbifID", "eventDate", "stateProvince", "decimalLatitude", "decimalLongitude",
                     "coordinateUncertaintyInMeters", "individualCount", "occurrenceStatus"]],
                "preview_raw_gbif", "RAW  gbif_ebird_sandhill_raw.csv (first rows)")
    table_image(obs[["date", "stateProvince", "lat", "lon", "count", "season", "count_class", "log10_count", "day_norm"]],
                "preview_clean_observations", "CLEAN  observations_clean.csv (first rows)")
    for path, name, cols, title in (
            (C.F_NCEI_RAW, "preview_raw_weather", ["site", "STATION", "DATE", "TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"],
             "RAW  ncei_daily_weather_raw.csv (first rows)"),):
        try:
            w = pd.read_csv(path)
            table_image(w[[c for c in cols if c in w.columns]], name, title)
        except FileNotFoundError:
            print(f"  - skipped {name}: {path.name} not found")
    cols = [c for c in ["site", "date", "station_id", "temp_mean_c", "precip_mm", "snow_mm",
                        "om_snowfall_mm", "om_snow_depth_max_m", "om_wind_max_kmh",
                        "n_records", "total_count"] if c in sd.columns]
    table_image(sd[cols], "preview_clean_site_daily", "CLEAN  site_daily_clean.csv (first rows)")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    raw, obs, sd, cells = load()
    print("EDA figures")
    fig01(cells); fig02(cells); fig03(cells)
    fig04(obs); fig05(obs); fig06(obs); fig07(obs); fig08(obs); fig09(obs)
    fig10(sd); fig11(sd); fig12(sd); fig13(sd)
    fig14(raw); fig15(sd); fig16(); fig17(); fig18(); fig19(); fig20()
    fig21(sd); fig22(sd)
    previews(raw, obs, sd)
    n_fig = len([m for m in made if m.startswith("fig")])
    print(f"\n{n_fig} figures + {len(made) - n_fig} previews written to {C.FIGS}")
    if n_fig < 10:
        print("WARNING: fewer than 10 figures were produced - check the messages above.")
