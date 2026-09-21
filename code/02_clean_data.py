"""
02_clean_data.py  -  General cleaning and preparation (Module 1).

Reads the untouched files in data/raw/ and writes analysis-ready files to data/clean/.
Every filtering step is counted and saved to data/clean/cleaning_log.json so the
DataPrep_EDA page can report exactly what was removed and why.

Model-specific preparation (sampling, scaling, train/test splits, etc.) happens later,
inside each model's own script.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C

LOG: dict[str, dict] = {}
MAX_PLAUSIBLE_COUNT = 500_000        # one checklist cannot hold more birds than this
MAX_COORD_UNCERTAINTY_M = 10_000     # locations vaguer than 10 km are unusable for mapping
LAT_RANGE, LON_RANGE = (14, 72), (-170, -50)   # North America


def note(section: str, step: str, before: int, after: int) -> None:
    LOG.setdefault(section, {})[step] = {"rows_before": int(before), "rows_after": int(after),
                                         "removed": int(before - after)}
    print(f"  [{section}] {step}: {before:,} -> {after:,}  (-{before - after:,})")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ observations
def _basic_clean(df: pd.DataFrame, section: str, key: tuple = ("gbifID",)) -> pd.DataFrame:
    """Steps shared by the continental sample and the site-area pull."""
    n = len(df)

    df = df.drop_duplicates(subset=list(key)); note(section, "drop duplicate gbifID", n, len(df)); n = len(df)

    df["date"] = pd.to_datetime(df["eventDate"].astype(str).str[:10], errors="coerce")
    df = df.dropna(subset=["date", "decimalLatitude", "decimalLongitude"])
    note(section, "drop rows missing date or coordinates", n, len(df)); n = len(df)

    df = df[df["decimalLatitude"].between(*LAT_RANGE) & df["decimalLongitude"].between(*LON_RANGE)]
    note(section, "drop coordinates outside North America", n, len(df)); n = len(df)

    vague = df["coordinateUncertaintyInMeters"] > MAX_COORD_UNCERTAINTY_M   # NaN compares False
    df = df[~vague]
    note(section, f"drop coordinate uncertainty > {MAX_COORD_UNCERTAINTY_M:,} m", n, len(df)); n = len(df)

    # eBird "X" (present, not counted) arrives as a missing count. Count-based analysis needs a number.
    df = df.dropna(subset=["individualCount"])
    note(section, "drop presence-only records (no count)", n, len(df)); n = len(df)

    df = df[df["individualCount"] >= 1]
    note(section, "drop counts < 1", n, len(df)); n = len(df)

    df = df[df["individualCount"] <= MAX_PLAUSIBLE_COUNT]
    note(section, f"drop implausible counts > {MAX_PLAUSIBLE_COUNT:,}", n, len(df))
    return df.copy()


def clean_observations() -> pd.DataFrame:
    print("Cleaning GBIF / eBird observations")
    df = _basic_clean(pd.read_csv(C.F_GBIF_RAW, low_memory=False), "observations")

    df["stateProvince"] = df["stateProvince"].fillna("Unknown").astype(str)
    df["individualCount"] = df["individualCount"].astype(int)

    # ---- derived variables (discretization + calendar features)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.dayofyear
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["season"] = df["month"].map(C.season_of)
    df["phase"] = df["month"].map(C.phase_of)
    df["era"] = np.where(df["year"] <= 2017, "2010-2017", "2018-2024")
    df["log10_count"] = np.log10(df["individualCount"])
    df["count_class"] = pd.cut(df["individualCount"], [0, 9, 99, 999, np.inf],
                               labels=["1-9", "10-99", "100-999", "1000+"]).astype(str)
    p99 = df["individualCount"].quantile(0.99)
    df["large_flock"] = (df["individualCount"] >= p99).astype(int)
    LOG["observations"]["large_flock_threshold_p99"] = float(p99)

    # ---- min-max normalised copies (0-1) for distance-based models later on
    for col, new in (("decimalLatitude", "lat_norm"), ("decimalLongitude", "lon_norm"), ("day", "day_norm")):
        lo, hi = df[col].min(), df[col].max()
        df[new] = (df[col] - lo) / (hi - lo)

    keep = ["gbifID", "date", "year", "month", "day", "week", "season", "phase", "era",
            "countryCode", "stateProvince", "decimalLatitude", "decimalLongitude",
            "individualCount", "log10_count", "count_class", "large_flock",
            "lat_norm", "lon_norm", "day_norm"]
    df = df[keep].rename(columns={"decimalLatitude": "lat", "decimalLongitude": "lon",
                                  "individualCount": "count"}).sort_values("date").reset_index(drop=True)
    assert df.isna().sum().sum() == 0, "NaNs remain in observations_clean"
    df.to_csv(C.F_OBS_CLEAN, index=False)
    print(f"  wrote {len(df):,} rows -> {C.F_OBS_CLEAN}")
    return df


def clean_site_observations() -> pd.DataFrame:
    """Every record inside the box around each reference site, assigned to its nearest site."""
    print("Cleaning site-area observations")
    df = _basic_clean(pd.read_csv(C.F_GBIF_SITES_RAW, low_memory=False), "site_observations",
                      key=("site", "gbifID"))       # the boxes of the two Platte sites overlap
    sites = pd.read_csv(C.F_SITES)
    dist = np.column_stack([haversine_km(df["decimalLatitude"].values, df["decimalLongitude"].values, r.lat, r.lon)
                            for r in sites.itertuples()])
    nearest = sites["site"].values[dist.argmin(axis=1)]
    within = dist.min(axis=1) <= C.SITE_RADIUS_KM
    n = len(df)
    df = df[within & (nearest == df["site"].values)]
    note("site_observations",
         f"keep records within {C.SITE_RADIUS_KM} km of their nearest reference site (removes box corners and overlap copies)",
         n, len(df))
    df = df.assign(count=df["individualCount"].astype(int))[
        ["site", "gbifID", "date", "decimalLatitude", "decimalLongitude", "count"]].rename(
        columns={"decimalLatitude": "lat", "decimalLongitude": "lon"}).sort_values(["site", "date"]).reset_index(drop=True)
    assert df.isna().sum().sum() == 0 and not df["gbifID"].duplicated().any()
    df.to_csv(C.F_SITE_OBS_CLEAN, index=False)
    print(f"  wrote {len(df):,} rows -> {C.F_SITE_OBS_CLEAN}")
    return df


# ------------------------------------------------------------------ weather
NCEI_OPTIONAL = {"SNOW": "snow_mm", "SNWD": "snow_depth_mm", "AWND": "wind_avg_ms"}
OPTIONAL_MIN_COVERAGE = 0.85         # keep an optional variable only if EVERY site has at least this share of days filled


def clean_ncei() -> pd.DataFrame:
    """NOAA NCEI GHCN-Daily station observations (metric units: deg C, mm, m/s)."""
    print("Cleaning NOAA NCEI weather")
    w = pd.read_csv(C.F_NCEI_RAW, low_memory=False)
    n = len(w)
    w["date"] = pd.to_datetime(w["DATE"], errors="coerce")
    w = w.dropna(subset=["date"]).drop_duplicates(subset=["site", "date"])
    note("weather_ncei", "drop bad dates and duplicate station-days", n, len(w)); n = len(w)
    for col in C.NCEI_DATATYPES:
        if col not in w.columns:
            w[col] = np.nan
    w = w.sort_values(["site", "date"]).reset_index(drop=True)

    # Snow cannot fall in June-September in these regions, so a blank snow value then means zero.
    summer = w["date"].dt.month.between(6, 9)
    for col in ("SNOW", "SNWD"):
        w.loc[summer & w[col].isna(), col] = 0.0
    # Short gaps (up to 3 days) inside a station's record are filled by linear interpolation.
    for col in NCEI_OPTIONAL:
        w[col] = w.groupby("site")[col].transform(lambda x: x.interpolate(limit=3, limit_area="inside"))

    w = w.dropna(subset=C.NCEI_CORE)
    note("weather_ncei", "drop days missing TMAX, TMIN or PRCP", n, len(w)); n = len(w)

    ok = ((w["TMAX"] >= w["TMIN"]) & w["TMAX"].between(-60, 60) & w["TMIN"].between(-70, 60)
          & (w["PRCP"] >= 0) & ~(w["SNOW"] < 0) & ~(w["SNWD"] < 0) & ~(w["AWND"] < 0))
    w = w[ok]
    note("weather_ncei", "drop physically impossible values", n, len(w)); n = len(w)

    out = pd.DataFrame({"site": w["site"], "date": w["date"], "station_id": w["STATION"],
                        "weather_source": "NOAA NCEI GHCN-Daily",
                        "temp_max_c": w["TMAX"], "temp_min_c": w["TMIN"],
                        "temp_mean_c": (w["TMAX"] + w["TMIN"]) / 2,      # TAVG is missing at many stations
                        "precip_mm": w["PRCP"]})
    kept, dropped = [], []
    for raw, new in NCEI_OPTIONAL.items():
        cover = float(w.groupby("site")[raw].apply(lambda x: x.notna().mean()).min())   # worst site decides
        if cover >= OPTIONAL_MIN_COVERAGE:
            out[new] = w[raw]
            kept.append(f"{raw}->{new} ({cover:.0%})")
        else:
            dropped.append(f"{raw} ({cover:.0%})")
    LOG["weather_ncei"]["optional_variables_kept"] = ", ".join(kept) or "none"
    LOG["weather_ncei"]["optional_variables_dropped_low_coverage"] = ", ".join(dropped) or "none"
    print(f"  optional variables kept: {kept}; dropped (coverage < {OPTIONAL_MIN_COVERAGE:.0%}): {dropped}")
    before = len(out)
    out = out.dropna()
    note("weather_ncei", "drop days with gaps in kept optional variables", before, len(out))
    return out.reset_index(drop=True)


def clean_openmeteo() -> pd.DataFrame | None:
    """Open-Meteo Historical Weather API (modelled ERA5 reanalysis): snowfall, snow depth and wind."""
    if not C.F_OPENMETEO_RAW.exists():
        print("Open-Meteo file not found - skipping (run 01_gather_data.py --only openmeteo).")
        return None
    print("Cleaning Open-Meteo weather")
    o = pd.read_csv(C.F_OPENMETEO_RAW)
    n = len(o)
    o["date"] = pd.to_datetime(o["time"], errors="coerce")
    o = o.dropna(subset=["date"]).drop_duplicates(subset=["site", "date"])
    note("weather_openmeteo", "drop bad dates and duplicate site-days", n, len(o)); n = len(o)
    has_depth = "snow_depth_max_m" in o.columns          # hourly snow depth is pulled separately (tighter free limit)
    need = ["temperature_2m_mean", "snowfall_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
            "wind_direction_10m_dominant"] + (["snow_depth_max_m"] if has_depth else [])
    o = o.dropna(subset=need)
    note("weather_openmeteo", "drop days with any missing value", n, len(o)); n = len(o)
    ok = ((o["snowfall_sum"] >= 0) & (o["wind_speed_10m_max"] >= 0)
          & (o["wind_gusts_10m_max"] >= o["wind_speed_10m_max"]) & o["temperature_2m_mean"].between(-60, 60)
          & o["wind_direction_10m_dominant"].between(0, 360))
    if has_depth:
        ok &= o["snow_depth_max_m"] >= 0
    o = o[ok]
    note("weather_openmeteo", "drop physically impossible values", n, len(o))
    out = pd.DataFrame({"site": o["site"], "date": o["date"],
                        "om_temp_mean_c": o["temperature_2m_mean"],
                        "om_snowfall_mm": o["snowfall_sum"] * 10,          # API gives cm of snowfall
                        "om_wind_max_kmh": o["wind_speed_10m_max"],
                        "om_gust_max_kmh": o["wind_gusts_10m_max"],
                        "om_wind_dir_deg": o["wind_direction_10m_dominant"]})
    if has_depth:
        out["om_snow_depth_max_m"] = o["snow_depth_max_m"]
    LOG["weather_openmeteo"]["snow_depth_included"] = bool(has_depth)
    return out.reset_index(drop=True)


def build_site_daily(site_obs: pd.DataFrame, ncei: pd.DataFrame, om: pd.DataFrame | None = None) -> pd.DataFrame:
    print("Building site-day table")
    sites = pd.read_csv(C.F_SITES)
    base = ncei
    if om is not None:
        base = ncei.merge(om, on=["site", "date"], how="inner")
        note("site_daily", "keep only days with both NCEI station weather and Open-Meteo values", len(ncei), len(base))

    # one row per site and day with the eBird reports made within the site radius that day
    agg = (site_obs.groupby(["site", "date"])
                   .agg(n_records=("count", "size"), total_count=("count", "sum"), max_count=("count", "max"))
                   .reset_index())

    matched = agg.merge(base[["site", "date"]], on=["site", "date"]).shape[0]
    note("site_daily", "drop site-days with crane reports but no usable weather (kept only where both exist)", len(agg), matched)
    sd = base.merge(agg, on=["site", "date"], how="left")
    for c in ("n_records", "total_count", "max_count"):
        sd[c] = sd[c].fillna(0).astype(int)        # no eBird record on that day = 0 records (not "missing")
    sd = sd.merge(sites[["site", "state", "lat", "lon", "role"]], on="site", how="left")

    sd["year"], sd["month"], sd["day"] = sd["date"].dt.year, sd["date"].dt.month, sd["date"].dt.dayofyear
    sd["season"] = sd["month"].map(C.season_of)
    sd["has_records"] = (sd["n_records"] > 0).astype(int)
    sd["temp_range_c"] = sd["temp_max_c"] - sd["temp_min_c"]
    sd = sd.sort_values(["site", "date"])
    sd["temp_7d_mean_c"] = sd.groupby("site")["temp_mean_c"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    sd = sd.reset_index(drop=True)
    assert sd.isna().sum().sum() == 0, "NaNs remain in site_daily_clean"
    sd.to_csv(C.F_SITE_DAILY_CLEAN, index=False)
    print(f"  wrote {len(sd):,} rows -> {C.F_SITE_DAILY_CLEAN}")
    return sd



# ------------------------------------------------------------------ land cover and effort
LC_START, LC_END = 2011, 2021                    # NLCD epochs compared (closest to the early and late bird windows)
EARLY, LATE = (2010, 2014), (2020, 2024)         # bird-report windows
MIN_CELL_COVER = 0.95                            # a cell needs at least 95% of its pixels inside the clip
MIN_CELL_REPORTS = 5                             # cells with fewer crane reports than this carry no signal


def _cell_index(x, y, cx, cy):
    """Grid cell of projected coordinates (metres), anchored on the site centre exactly as in gather_landcover()."""
    half = C.NLCD_CLIP_KM * 1000
    n_side = int(2 * half / C.CELL_M) + 1
    i = np.floor((x - (cx - half)) / C.CELL_M).astype(int)
    j = np.floor((y - (cy - half)) / C.CELL_M).astype(int)
    return j * n_side + i


def clean_landcover(site_obs: pd.DataFrame) -> pd.DataFrame:
    """NLCD pixel counts -> land-cover shares per site and 5 km cell, joined to crane reports per cell."""
    from pyproj import Transformer
    print("Cleaning NLCD land cover")
    raw = pd.read_csv(C.F_NLCD_RAW)
    sites = pd.read_csv(C.F_SITES).set_index("site")
    n = len(raw)
    raw = raw[raw["nlcd_class"].isin([c for v in C.NLCD_GROUPS.values() for c in v] + [12])]
    note("landcover", "drop pixel classes outside the 16 NLCD codes used (no-data)", n, len(raw)); n = len(raw)

    class_to_group = {c: g for g, cs in C.NLCD_GROUPS.items() for c in cs}
    raw["group"] = raw["nlcd_class"].map(class_to_group).fillna("other")
    cell = (raw.groupby(["site", "year", "cell_id", "cell_lat", "cell_lon", "group"])["pixels"].sum()
               .unstack("group", fill_value=0).reset_index())
    groups = list(C.NLCD_GROUPS) + ["other"]
    for g in groups:
        if g not in cell.columns:
            cell[g] = 0
    cell["total_px"] = cell[groups].sum(axis=1)
    full = (C.CELL_M / 30) ** 2
    before = len(cell)
    cell = cell[cell["total_px"] >= MIN_CELL_COVER * full]
    note("landcover", "drop partial cells at the edge of a clip (< 95% of pixels)", before, len(cell)); before = len(cell)
    d = haversine_km(cell["cell_lat"].values, cell["cell_lon"].values,
                     sites.loc[cell["site"], "lat"].values, sites.loc[cell["site"], "lon"].values)
    cell = cell[d <= C.SITE_RADIUS_KM].copy()
    note("landcover", f"keep cells whose centre is within {C.SITE_RADIUS_KM} km of the site", before, len(cell))
    for g in groups:
        cell[f"pct_{g}"] = 100 * cell[g] / cell["total_px"]
    keep = ["site", "year", "cell_id", "cell_lat", "cell_lon"] + [f"pct_{g}" for g in groups]
    cell_year = cell[keep].sort_values(["site", "year", "cell_id"]).reset_index(drop=True)
    assert cell_year.isna().sum().sum() == 0
    cell_year.to_csv(C.F_LANDCOVER_CLEAN, index=False)
    print(f"  wrote {len(cell_year):,} rows -> {C.F_LANDCOVER_CLEAN}")

    site_year = (cell.groupby(["site", "year"])[groups + ["total_px"]].sum().reset_index())
    for g in groups:
        site_year[f"pct_{g}"] = 100 * site_year[g] / site_year["total_px"]
    site_year = site_year[["site", "year"] + [f"pct_{g}" for g in groups]]
    site_year.to_csv(C.F_SITE_LANDCOVER_CLEAN, index=False)
    print(f"  wrote {len(site_year):,} rows -> {C.F_SITE_LANDCOVER_CLEAN}")

    # ---- crane reports per cell in the early and late windows
    to_albers = Transformer.from_crs(4326, 5070, always_xy=True)
    centres = {name: to_albers.transform(r.lon, r.lat) for name, r in sites.iterrows()}
    o = site_obs.copy()
    xy = np.array([to_albers.transform(lo, la) for lo, la in zip(o["lon"], o["lat"])])
    ids = np.zeros(len(o), dtype=int)
    for nm, (cx, cy) in centres.items():
        m = (o["site"] == nm).values
        ids[m] = _cell_index(xy[m, 0], xy[m, 1], cx, cy)
    o["cell_id"] = ids
    o["window"] = np.where(o["date"].dt.year.between(*EARLY), "early",
                           np.where(o["date"].dt.year.between(*LATE), "late", "middle"))
    o = o[o["window"] != "middle"]
    rep = (o.groupby(["site", "cell_id", "window"]).agg(reports=("count", "size"), cranes=("count", "sum"))
             .unstack("window", fill_value=0))
    rep.columns = [f"{a}_{b}" for a, b in rep.columns]
    rep = rep.reset_index()
    for c in ("reports_early", "reports_late", "cranes_early", "cranes_late"):
        if c not in rep.columns:
            rep[c] = 0

    # ---- eBird hotspots per cell (a fixed indicator of where observers go)
    hs = pd.read_csv(C.F_HOTSPOTS_CLEAN)
    hs_xy = np.array([to_albers.transform(lo, la) for lo, la in zip(hs["lng"], hs["lat"])])
    hd = np.column_stack([haversine_km(hs["lat"].values, hs["lng"].values, r.lat, r.lon) for r in sites.itertuples()])
    hs["site"] = sites.index.values[hd.argmin(axis=1)]
    hs = hs[hd.min(axis=1) <= C.SITE_RADIUS_KM].copy()
    hs_ids = np.zeros(len(hs), dtype=int)
    hxy = hs_xy[hd.min(axis=1) <= C.SITE_RADIUS_KM]
    for nm, (cx, cy) in centres.items():
        m = (hs["site"] == nm).values
        hs_ids[m] = _cell_index(hxy[m, 0], hxy[m, 1], cx, cy)
    hs["cell_id"] = hs_ids
    hot = hs.groupby(["site", "cell_id"]).size().rename("hotspots").reset_index()

    # ---- one row per cell: land cover at both epochs, change, and crane share early vs late
    a = cell_year[cell_year["year"] == LC_START].set_index(["site", "cell_id"])
    b = cell_year[cell_year["year"] == LC_END].set_index(["site", "cell_id"])
    both = a.join(b, lsuffix=f"_{LC_START}", rsuffix=f"_{LC_END}", how="inner")
    out = both[[f"cell_lat_{LC_START}", f"cell_lon_{LC_START}"]].copy()
    out.columns = ["cell_lat", "cell_lon"]
    for g in groups:
        out[f"pct_{g}_{LC_START}"] = both[f"pct_{g}_{LC_START}"]
        out[f"pct_{g}_{LC_END}"] = both[f"pct_{g}_{LC_END}"]
        out[f"d_{g}_pp"] = both[f"pct_{g}_{LC_END}"] - both[f"pct_{g}_{LC_START}"]
    out = out.reset_index().merge(rep, on=["site", "cell_id"], how="left").merge(hot, on=["site", "cell_id"], how="left")
    for c in ("reports_early", "reports_late", "cranes_early", "cranes_late", "hotspots"):
        out[c] = out[c].fillna(0).astype(int)                # a cell with no report is zero reports, not missing
    tot = out.groupby("site")[["reports_early", "reports_late"]].transform("sum")
    out["share_early_pct"] = 100 * out["reports_early"] / tot["reports_early"]
    out["share_late_pct"] = 100 * out["reports_late"] / tot["reports_late"]
    out["d_share_pp"] = out["share_late_pct"] - out["share_early_pct"]
    # every cell stays in the file; cells with too few crane reports are flagged rather than deleted, because
    # cells with no reports are needed to ask where cranes were NOT reported
    out["enough_reports"] = ((out["reports_early"] + out["reports_late"]) >= MIN_CELL_REPORTS).astype(int)
    LOG["landcover"]["cells_flagged_enough_reports"] = (
        f"{int(out['enough_reports'].sum()):,} of {len(out):,} cells have at least {MIN_CELL_REPORTS} crane reports")
    out = out.sort_values(["site", "cell_id"]).reset_index(drop=True)
    assert out.isna().sum().sum() == 0
    out.to_csv(C.F_CELL_CHANGE_CLEAN, index=False)
    print(f"  wrote {len(out):,} rows -> {C.F_CELL_CHANGE_CLEAN}")
    return out


def clean_effort() -> pd.DataFrame:
    """Crane records per 1,000 eBird records of any species, by site and year."""
    print("Building reporting-effort table")
    eff = pd.read_csv(C.F_EFFORT_RAW)
    crane = (pd.read_csv(C.F_GBIF_SITES_RAW, low_memory=False, usecols=["site", "gbifID", "year"])
               .drop_duplicates(["site", "gbifID"]).groupby(["site", "year"]).size().rename("crane_records").reset_index())
    out = eff.merge(crane, on=["site", "year"], how="left")
    out["crane_records"] = out["crane_records"].fillna(0).astype(int)
    out["crane_per_1000"] = 1000 * out["crane_records"] / out["all_species_records"]
    note("effort", "site-years kept (no rows removed)", len(eff), len(out))
    assert out.isna().sum().sum() == 0
    out.to_csv(C.F_EFFORT_CLEAN, index=False)
    print(f"  wrote {len(out):,} rows -> {C.F_EFFORT_CLEAN}")
    return out


# ------------------------------------------------------------------ eBird API extracts
def clean_ebird_api() -> None:
    if C.F_EBIRD_API_RAW.exists():
        print("Cleaning eBird API recent observations")
        d = pd.read_csv(C.F_EBIRD_API_RAW)
        n = len(d)
        d = d.drop_duplicates(subset=[c for c in ("subId", "locId", "obsDt") if c in d.columns]); note("ebird_api", "drop duplicates", n, len(d)); n = len(d)
        d["obsDt"] = pd.to_datetime(d["obsDt"], errors="coerce")
        d = d.dropna(subset=["obsDt", "lat", "lng", "howMany"]); note("ebird_api", "drop missing date/coords/count", n, len(d)); n = len(d)
        d = d[(d["howMany"] >= 1) & (d["howMany"] <= MAX_PLAUSIBLE_COUNT)]; note("ebird_api", "drop implausible counts", n, len(d))
        d["howMany"] = d["howMany"].astype(int)
        d["state"] = d["regionQuery"].str[-2:]
        keep = [c for c in ("subId", "locId", "locName", "obsDt", "howMany", "lat", "lng", "state", "pulled_on") if c in d.columns]
        d = d[keep]
        assert d.isna().sum().sum() == 0
        d.to_csv(C.F_EBIRD_API_CLEAN, index=False)
        print(f"  wrote {len(d):,} rows -> {C.F_EBIRD_API_CLEAN}")
    else:
        print("No eBird API raw file found - skipping (requires EBIRD_API_KEY).")

    if C.F_HOTSPOTS_RAW.exists():
        print("Cleaning eBird hotspots")
        h = pd.read_csv(C.F_HOTSPOTS_RAW)
        n = len(h)
        h = h.dropna(subset=["locId", "lat", "lng"]) if "lat" in h.columns else h.dropna(subset=["locId", "latitude", "longitude"])
        h = h.rename(columns={"latitude": "lat", "longitude": "lng"})
        if "numSpeciesAllTime" in h.columns:
            h["numSpeciesAllTime"] = h["numSpeciesAllTime"].fillna(0).astype(int)
        h = h.drop(columns=[c for c in ("lastObsDt", "latestObsDt") if c in h.columns])
        note("hotspots", "drop rows missing id/coords", n, len(h)); n = len(h)
        h = h.dropna()
        note("hotspots", "drop rows with any other missing field (county or checklist total)", n, len(h))
        if "numChecklistsAllTime" in h.columns:
            h["numChecklistsAllTime"] = h["numChecklistsAllTime"].astype(int)
        h.to_csv(C.F_HOTSPOTS_CLEAN, index=False)
        print(f"  wrote {len(h):,} rows -> {C.F_HOTSPOTS_CLEAN}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    obs = clean_observations()
    site_obs = clean_site_observations() if C.F_GBIF_SITES_RAW.exists() else None
    if site_obs is None:
        raise SystemExit("Site-area records not found. Run 01_gather_data.py --only gbif_sites first.")
    ncei = clean_ncei() if C.F_NCEI_RAW.exists() else None
    if ncei is None:
        raise SystemExit("No weather data found. Run 01_gather_data.py --only ncei first.")
    om = clean_openmeteo()
    build_site_daily(site_obs, ncei, om)
    clean_ebird_api()
    if C.F_NLCD_RAW.exists() and C.F_EFFORT_RAW.exists():
        clean_landcover(site_obs)
        clean_effort()
    else:
        print("Land-cover files not found - skipping (run 01_gather_data.py --only landcover and --only effort).")
    (C.CLEAN / "cleaning_log.json").write_text(json.dumps(LOG, indent=2))
    print("\nCleaning log saved to data/clean/cleaning_log.json")
