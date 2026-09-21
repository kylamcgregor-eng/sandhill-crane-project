"""
01_gather_data.py  -  Data gathering for the sandhill crane project.

Sources (all pulled through web APIs with the `requests` package):

  1. GBIF Occurrence API   (no key)  eBird sightings of Sandhill Crane, 2010-2024
                                     (a random sample across the continent + every record near each reference site)
       https://api.gbif.org/v1/occurrence/search
  2. eBird API 2.0         (free key) recent (last 30 days) sightings + hotspot lists
       https://api.ebird.org/v2/data/obs/{regionCode}/recent/sancra
  3. NOAA NCEI GHCN-Daily  (no key)  daily station weather (temperature, precipitation, snow)
       https://www.ncei.noaa.gov/access/services/data/v1   (station lists: /pub/data/ghcn/daily/)
  4. Open-Meteo archive API  (no key) modelled daily snowfall, snow depth and wind
       https://archive-api.open-meteo.com/v1/archive
  5. USGS NLCD land cover     (no key)  30 m land-cover clips around each site, MRLC Web Coverage Service
       https://www.mrlc.gov/geoserver/mrlc_download/wcs
  6. A small hand-built table of reference sites (created dataset).

Usage
-----
    pip install -r requirements.txt
    export EBIRD_API_KEY="your-key"        # optional; step 2 is skipped without it
    python code/01_gather_data.py            # everything
    python code/01_gather_data.py --only gbif|gbif_sites|ebird|ncei|openmeteo|landcover|effort|sites
    python code/01_gather_data.py --quick    # 2020-2024 only, 300 records per cell

Raw files are written to data/raw/ and are never edited by later scripts.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import pandas as pd
import requests

import config as C

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CSCI5612-sandhill-crane-project (student coursework)"})
HARD_LIMIT_S = 90          # wall-clock cap per request (a server that trickles bytes never trips `timeout`)


def _fetch(url, params, headers):
    """One request on a fresh connection (no keep-alive reuse); returns (status, body-bytes)."""
    with requests.Session() as sess:
        sess.headers.update(SESSION.headers)
        sess.headers["Connection"] = "close"
        r = sess.get(url, params=params, headers=headers, timeout=(15, 45))
        return r.status_code, r.content


# ------------------------------------------------------------------ helpers
def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             retries: int = 8, pause: float = 0.15):
    """GET with retry/backoff and a hard wall-clock limit. Returns parsed JSON."""
    for attempt in range(retries):
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            status, body = pool.submit(_fetch, url, params, headers).result(timeout=HARD_LIMIT_S)
            if status == 429 or status >= 500:
                raise requests.HTTPError(f"HTTP {status}")
            if status >= 400:
                raise requests.HTTPError(f"HTTP {status} for {url}")
            time.sleep(pause)
            return json.loads(body)
        except (requests.RequestException, ValueError, FuturesTimeout) as exc:
            wait = min(2 ** attempt, 20)
            print(f"  ! {type(exc).__name__}: {exc} (attempt {attempt + 1}/{retries}); retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
        finally:
            pool.shutdown(wait=False)      # never wait on a stuck worker thread
    raise RuntimeError(f"Giving up on {url} {params}")


# ------------------------------------------------------------------ 1. GBIF / eBird
def resolve_taxon_key() -> int:
    """Endpoint: GET /species/match?name=Antigone canadensis"""
    js = get_json(f"{C.GBIF_API}/species/match", {"name": C.SPECIES_NAME})
    key = js.get("usageKey")
    if not key:
        raise RuntimeError(f"GBIF could not match {C.SPECIES_NAME}: {js}")
    print(f"GBIF taxonKey for {js.get('scientificName')}: {key}")
    return int(key)


def resolve_dataset_key() -> str:
    """Confirm the eBird dataset key; fall back to a search if it does not check out."""
    try:
        js = get_json(f"{C.GBIF_API}/dataset/{C.EBIRD_DATASET_KEY}")
        if "ebird" in js.get("title", "").lower():
            print(f"GBIF dataset: {js['title']}")
            return C.EBIRD_DATASET_KEY
    except RuntimeError:
        pass
    js = get_json(f"{C.GBIF_API}/dataset/search", {"q": "eBird Observation Dataset", "type": "OCCURRENCE"})
    for res in js.get("results", []):
        if "ebird" in res.get("title", "").lower():
            print(f"GBIF dataset (via search): {res['title']} [{res['key']}]")
            return res["key"]
    raise RuntimeError("Could not locate the eBird dataset on GBIF.")


KEEP = ["gbifID", "eventDate", "year", "month", "day", "decimalLatitude", "decimalLongitude",
        "coordinateUncertaintyInMeters", "countryCode", "stateProvince", "locality",
        "individualCount", "occurrenceStatus", "basisOfRecord", "scientificName",
        "taxonKey", "datasetKey"]


def fetch_cell(taxon: int, dataset: str, country: str, year: int, month: int,
               cap: int, rng: random.Random) -> list[dict]:
    """
    One (country, year, month) cell.

    Endpoint example (core GET request):
      https://api.gbif.org/v1/occurrence/search?taxonKey=<key>&datasetKey=<eBird>&country=US
          &year=2018&month=3&hasCoordinate=true&occurrenceStatus=PRESENT&limit=300&offset=0

    GBIF caps deep paging at offset+limit <= 100,000. If a cell holds more records than
    `cap`, several slices at random offsets are read instead of every record.
    """
    base = {"taxonKey": taxon, "datasetKey": dataset, "country": country, "year": year,
            "month": month, "hasCoordinate": "true", "occurrenceStatus": "PRESENT"}
    total = get_json(f"{C.GBIF_API}/occurrence/search", {**base, "limit": 0})["count"]
    if total == 0:
        return []
    reachable = min(total, 100_000)
    n_slices = max(1, cap // C.SLICE)
    if reachable <= cap:
        offsets = list(range(0, reachable, C.SLICE))
    else:
        max_start = max(0, reachable - C.SLICE)
        offsets = sorted({rng.randint(0, max_start) for _ in range(n_slices)})
    rows: list[dict] = []
    for off in offsets:
        js = get_json(f"{C.GBIF_API}/occurrence/search", {**base, "limit": C.SLICE, "offset": off})
        for rec in js.get("results", []):
            row = {k: rec.get(k) for k in KEEP}
            row["cell_total_records"] = total          # keeps sampling effort visible
            rows.append(row)
    return rows


def gather_gbif(quick: bool = False, countries: list[str] | None = None,
                year_range: tuple[int, int] | None = None) -> None:
    years = [y for y in C.YEARS if y >= 2020] if quick else C.YEARS
    if year_range:
        years = [y for y in years if year_range[0] <= y <= year_range[1]]
    cap = C.SLICE if quick else C.RECORDS_PER_CELL
    taxon, dataset = resolve_taxon_key(), resolve_dataset_key()

    cache = C.RAW / ".gbif_cells.jsonl"              # lets an interrupted run resume
    done: dict[str, list[dict]] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:                        # a run killed mid-write leaves a partial last line
                continue
            done[rec["cell"]] = rec["rows"]

    for country in countries or C.COUNTRIES:
        for year in years:
            for month in range(1, 13):
                cell = f"{country}-{year}-{month:02d}"
                if cell in done:
                    continue
                # one seeded generator per cell: the sampled offsets do not depend on run order or resumes
                rng = random.Random(f"{C.RANDOM_SEED}-{cell}")
                rows = fetch_cell(taxon, dataset, country, year, month, cap, rng)
                done[cell] = rows
                with cache.open("a") as fh:
                    fh.write(json.dumps({"cell": cell, "rows": rows}) + "\n")
                print(f"  {cell}: {len(rows)} records")

    df = pd.DataFrame([r for rows in done.values() for r in rows])
    df.to_csv(C.F_GBIF_RAW, index=False)
    print(f"Wrote {len(df):,} raw GBIF/eBird rows -> {C.F_GBIF_RAW}")



def gather_gbif_sites(only_sites: list[str] | None = None) -> None:
    """
    Every eBird record of the species inside a box around each reference site (about
    SITE_RADIUS_KM in each direction), one request series per site and year. Unlike the
    random sample above this is a complete pull, so daily counts at a site are real totals.

    Endpoint example:
      https://api.gbif.org/v1/occurrence/search?taxonKey=<key>&datasetKey=<eBird>&year=2018
          &decimalLatitude=40.16,41.24&decimalLongitude=-99.70,-98.46&hasCoordinate=true
          &occurrenceStatus=PRESENT&limit=300&offset=0
    """
    import math
    taxon, dataset = resolve_taxon_key(), resolve_dataset_key()
    cache = C.RAW / ".gbif_site_cells.jsonl"
    done: dict[str, list[dict]] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            done[rec["cell"]] = rec["rows"]

    for name, _state, lat, lon, _role in C.SITES:
        if only_sites and not any(t.lower() in name.lower() for t in only_sites):
            continue
        dlat = C.SITE_RADIUS_KM / 111.0
        dlon = C.SITE_RADIUS_KM / (111.0 * math.cos(math.radians(lat)))
        box = {"decimalLatitude": f"{lat - dlat:.4f},{lat + dlat:.4f}",
               "decimalLongitude": f"{lon - dlon:.4f},{lon + dlon:.4f}"}
        for year in C.YEARS:
            cell = f"{name}|{year}"
            if cell in done:
                continue
            base = {"taxonKey": taxon, "datasetKey": dataset, "year": year, "hasCoordinate": "true",
                    "occurrenceStatus": "PRESENT", **box}
            rows, offset = [], 0
            while True:
                js = get_json(f"{C.GBIF_API}/occurrence/search", {**base, "limit": C.SLICE, "offset": offset})
                for rec in js.get("results", []):
                    row = {k: rec.get(k) for k in KEEP}
                    row["site"] = name
                    rows.append(row)
                offset += C.SLICE
                if js.get("endOfRecords", True) or not js.get("results"):
                    break
            done[cell] = rows
            with cache.open("a") as fh:
                fh.write(json.dumps({"cell": cell, "rows": rows}) + "\n")
            print(f"  {name} {year}: {len(rows)} records")

    df = pd.DataFrame([r for rows in done.values() for r in rows])
    df.to_csv(C.F_GBIF_SITES_RAW, index=False)
    print(f"Wrote {len(df):,} raw site-area rows -> {C.F_GBIF_SITES_RAW}")


# ------------------------------------------------------------------ 2. eBird API 2.0
def gather_ebird_api() -> None:
    """
    Core endpoints used (header  X-eBirdApiToken: <key>):
      GET https://api.ebird.org/v2/data/obs/US-NE/recent/sancra?back=30&detail=full&maxResults=10000
      GET https://api.ebird.org/v2/ref/hotspot/US-CO?fmt=json&back=30
    The recent-observations endpoint only reaches back 30 days, so the file is
    APPENDED to on every run (each row carries `pulled_on`), which builds a time
    series over the semester.
    """
    key = os.environ.get("EBIRD_API_KEY")
    if not key:
        print("EBIRD_API_KEY not set - skipping eBird API step "
              "(get a free key at https://ebird.org/api/keygen).")
        return
    headers = {"X-eBirdApiToken": key}
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    frames = []
    for region in C.EBIRD_REGIONS:
        url = f"{C.EBIRD_API}/data/obs/{region}/recent/{C.EBIRD_SPECIES_CODE}"
        js = get_json(url, {"back": 30, "detail": "full", "maxResults": 10000}, headers)
        print(f"  eBird recent {region}: {len(js)} observations")
        if js:
            d = pd.DataFrame(js)
            d["regionQuery"] = region
            frames.append(d)
    if frames:
        new = pd.concat(frames, ignore_index=True)
        new["pulled_on"] = today
        if C.F_EBIRD_API_RAW.exists():
            old = pd.read_csv(C.F_EBIRD_API_RAW)
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(C.F_EBIRD_API_RAW, index=False)
        print(f"Wrote {len(new):,} rows -> {C.F_EBIRD_API_RAW}")

    hs = []
    for region in C.EBIRD_HOTSPOT_REGIONS:
        js = get_json(f"{C.EBIRD_API}/ref/hotspot/{region}", {"fmt": "json"}, headers)
        d = pd.DataFrame(js)
        d["regionQuery"] = region
        hs.append(d)
        print(f"  eBird hotspots {region}: {len(d)}")
    if hs:
        pd.concat(hs, ignore_index=True).to_csv(C.F_HOTSPOTS_RAW, index=False)
        print(f"Wrote hotspots -> {C.F_HOTSPOTS_RAW}")


# ------------------------------------------------------------------ 3. NOAA NCEI GHCN-Daily
def get_text(url: str, params: dict | None = None, retries: int = 5, pause: float = 0.3) -> str:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            time.sleep(pause)
            return r.text
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  ! {exc} (attempt {attempt + 1}/{retries}); retrying in {wait}s", file=sys.stderr)
            SESSION.close()          # drop a possibly stuck connection before retrying
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} {params}")


def _cached_download(url: str, name: str) -> str:
    C.NCEI_CACHE.mkdir(parents=True, exist_ok=True)
    path = C.NCEI_CACHE / name
    if not path.exists():
        print(f"  downloading {url}")
        path.write_text(get_text(url), encoding="utf-8")
    return str(path)


def _haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def choose_ncei_stations() -> pd.DataFrame:
    """
    For every reference site, pick the best GHCN-Daily station:
      * must report TMAX, TMIN and PRCP from the first study year through the last one,
      * nearer is better; first-order weather stations (IDs starting USW) and stations that also
        report snow and wind get a small distance bonus.
    Station files (fixed width, documented in ghcnd-readme.txt):
      ghcnd-stations.txt   ID 1-11 | LAT 13-20 | LON 22-30 | ELEV 32-37 | STATE 39-40 | NAME 42-71
      ghcnd-inventory.txt  ID 1-11 | LAT 13-20 | LON 22-30 | ELEMENT 32-35 | FIRST 37-40 | LAST 42-45
    """
    st = pd.read_fwf(_cached_download(C.NCEI_STATIONS_URL, "ghcnd-stations.txt"),
                     colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)],
                     names=["id", "lat", "lon", "elev", "state", "name"], header=None,
                     dtype={"id": str, "state": str, "name": str})
    inv = pd.read_fwf(_cached_download(C.NCEI_INVENTORY_URL, "ghcnd-inventory.txt"),
                      colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
                      names=["id", "lat", "lon", "element", "first", "last"], header=None,
                      dtype={"id": str, "element": str})
    inv = inv[inv["id"].str.startswith("US")]
    y0, y1 = int(C.WEATHER_START[:4]), int(C.WEATHER_END[:4])
    full = inv[(inv["first"] <= y0) & (inv["last"] >= y1)]
    have = full.groupby("id")["element"].apply(set)                      # elements covering the whole period
    st = st[st["id"].isin(have.index)].copy()
    st["elements"] = st["id"].map(have)

    chosen = []
    for name, state, lat, lon, _role in C.SITES:
        pick = None
        for radius in C.NCEI_SEARCH_KM:
            cand = st[_haversine_km(lat, lon, st["lat"], st["lon"]) <= radius].copy()
            cand = cand[cand["elements"].apply(lambda e: set(C.NCEI_CORE) <= e)]
            if cand.empty:
                continue
            cand["dist_km"] = _haversine_km(lat, lon, cand["lat"], cand["lon"])
            cand["score"] = (cand["dist_km"]
                             - 20 * cand["id"].str.startswith("USW")
                             - 10 * cand["elements"].apply(lambda e: "SNOW" in e and "SNWD" in e)
                             - 10 * cand["elements"].apply(lambda e: "AWND" in e))
            pick = cand.sort_values("score").iloc[0]
            break
        if pick is None:
            print(f"  ! no complete GHCN-Daily station within {C.NCEI_SEARCH_KM[-1]} km of {name}; site skipped for NCEI")
            continue
        chosen.append({"site": name, "station_id": pick["id"], "station_name": str(pick["name"]).strip(),
                       "station_lat": pick["lat"], "station_lon": pick["lon"], "station_elev_m": pick["elev"],
                       "dist_km": round(float(pick["dist_km"]), 1), "elements": ",".join(sorted(pick["elements"]))})
        print(f"  {name}: {pick['id']} {str(pick['name']).strip()} ({pick['dist_km']:.0f} km)")
    return pd.DataFrame(chosen)


def gather_ncei() -> None:
    """
    Core endpoint (NCEI Access Data Service, no key):
      https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00014935
          &startDate=2010-01-01&endDate=2024-12-31&dataTypes=TMAX,TMIN,TAVG,PRCP,SNOW,SNWD,AWND
          &format=csv&units=metric&includeStationName=true
    Returns one row per station-day; missing observations are blank.
    """
    stations = choose_ncei_stations()
    if stations.empty:
        print("No stations selected - nothing to download.")
        return
    stations.to_csv(C.F_NCEI_STATIONS, index=False)
    print(f"Wrote station choices -> {C.F_NCEI_STATIONS}")

    frames = []
    for row in stations.itertuples():
        text = get_text(C.NCEI_DATA_API, {
            "dataset": "daily-summaries", "stations": row.station_id,
            "startDate": C.WEATHER_START, "endDate": C.WEATHER_END,
            "dataTypes": ",".join(C.NCEI_DATATYPES), "format": "csv", "units": "metric",
            "includeStationName": "true"})
        d = pd.read_csv(io.StringIO(text))
        d.insert(0, "site", row.site)
        frames.append(d)
        print(f"  NCEI {row.site} ({row.station_id}): {len(d):,} station-days")
    pd.concat(frames, ignore_index=True).to_csv(C.F_NCEI_RAW, index=False)
    print(f"Wrote NCEI weather -> {C.F_NCEI_RAW}")




# ------------------------------------------------------------------ 5b. Open-Meteo
def gather_openmeteo() -> None:
    """
    Open-Meteo Historical Weather API (no key; modelled reanalysis, not station readings).
    One request per site for the whole study period. Snow depth is only offered hourly, so the
    hourly values are summarized to the daily maximum here.
      https://archive-api.open-meteo.com/v1/archive?latitude=40.7&longitude=-99.08
          &start_date=2010-01-01&end_date=2024-12-31&timezone=auto
          &daily=temperature_2m_mean,snowfall_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant
          &hourly=snow_depth
    """
    cache = C.OM_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    slug = lambda n: n.lower().replace(" ", "_").replace("'", "")

    # (a) daily variables: one request per site (resumable; each finished site is cached)
    daily = []
    for name, _state, lat, lon, _role in C.SITES:
        f = cache / (slug(name) + "_daily.csv")
        if not f.exists():
            try:
                js = get_json(C.OPEN_METEO_ARCHIVE, {
                    "latitude": lat, "longitude": lon, "start_date": C.WEATHER_START, "end_date": C.WEATHER_END,
                    "daily": ",".join(C.OM_DAILY_VARS), "timezone": "auto"}, retries=4)
            except RuntimeError:
                print(f"  ! {name}: daily request failed; re-run this step later to resume", file=sys.stderr)
                continue
            d = pd.DataFrame(js["daily"])
            d.insert(0, "site", name)
            d["grid_lat"], d["grid_lon"] = js["latitude"], js["longitude"]
            d.to_csv(f, index=False)
            print(f"  Open-Meteo daily {name}: {len(d):,} days (grid point {js['latitude']:.2f}, {js['longitude']:.2f})")
            time.sleep(5)
        else:
            print(f"  Open-Meteo daily {name}: cached")
        daily.append(pd.read_csv(f))
    if len(daily) < len(C.SITES):
        print(f"  Only {len(daily)} of {len(C.SITES)} sites have daily weather; not writing the combined file yet.")
        return
    out = pd.concat(daily, ignore_index=True)

    # (b) hourly snow depth (a much larger request that the free service limits more tightly); daily maximum kept
    depth, stopped = [], False
    for name, _state, lat, lon, _role in C.SITES:
        f = cache / (slug(name) + "_snowdepth.csv")
        if not f.exists() and not stopped:
            try:
                js = get_json(C.OPEN_METEO_ARCHIVE, {
                    "latitude": lat, "longitude": lon, "start_date": C.WEATHER_START, "end_date": C.WEATHER_END,
                    "hourly": "snow_depth", "timezone": "auto"}, retries=2)
                h = pd.DataFrame(js["hourly"])
                h["time"] = pd.to_datetime(h["time"]).dt.strftime("%Y-%m-%d")
                dm = h.groupby("time")["snow_depth"].max().rename("snow_depth_max_m").reset_index()
                dm.insert(0, "site", name)
                dm.to_csv(f, index=False)
                print(f"  Open-Meteo snow depth {name}: {len(dm):,} days")
                time.sleep(5)
            except RuntimeError:
                stopped = True
                print("  ! snow depth request refused (free-service limit); stopping. Re-run this step later to resume.",
                      file=sys.stderr)
        if f.exists():
            depth.append(pd.read_csv(f))
    if len(depth) == len(C.SITES):
        out = out.merge(pd.concat(depth, ignore_index=True), on=["site", "time"], how="left")
    else:
        print(f"  Snow depth is available for {len(depth)} of {len(C.SITES)} sites, so it is left out of the combined file for now.")
    out.to_csv(C.F_OPENMETEO_RAW, index=False)
    print(f"Wrote Open-Meteo weather -> {C.F_OPENMETEO_RAW}")


# ------------------------------------------------------------------ 5. land cover (NLCD) and reporting effort
def gather_landcover() -> None:
    """
    USGS National Land Cover Database (30 m), from the MRLC Web Coverage Service. One small GeoTIFF is
    requested per site and NLCD year, then pixels are counted by class inside 5 km grid cells.

    Endpoint example (X/Y are metres in EPSG:5070, the coverage's native projection):
      https://www.mrlc.gov/geoserver/mrlc_download/wcs?service=WCS&version=2.0.1&request=GetCoverage
          &coverageId=mrlc_download__NLCD_2011_Land_Cover_L48&format=image/geotiff
          &subset=X(-273051,-213051)&subset=Y(1954576,2014576)
    """
    import warnings
    import numpy as np
    import tifffile
    from pyproj import Transformer

    warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
    to_albers = Transformer.from_crs(4326, 5070, always_xy=True)
    to_lonlat = Transformer.from_crs(5070, 4326, always_xy=True)
    half = C.NLCD_CLIP_KM * 1000
    out_rows = []
    for name, _state, lat, lon, _role in C.SITES:
        cx, cy = to_albers.transform(lon, lat)
        for year in C.NLCD_YEARS:
            params = {"service": "WCS", "version": "2.0.1", "request": "GetCoverage",
                      "coverageId": C.NLCD_COVERAGE.format(year=year), "format": "image/geotiff"}
            url = C.NLCD_WCS + "?" + "&".join(f"{k}={v}" for k, v in params.items()) + \
                f"&subset=X({cx - half:.0f},{cx + half:.0f})&subset=Y({cy - half:.0f},{cy + half:.0f})"
            for attempt in range(6):
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        resp = pool.submit(SESSION.get, url, timeout=(15, 90)).result(timeout=150)
                    resp.raise_for_status()
                    break
                except Exception as exc:                       # noqa: BLE001 - retry on any transient failure
                    print(f"  ! {type(exc).__name__}: {exc} (attempt {attempt + 1}/6)", file=sys.stderr)
                    time.sleep(min(2 ** attempt, 20))
            else:
                raise RuntimeError(f"Giving up on NLCD {name} {year}")
            with tifffile.TiffFile(io.BytesIO(resp.content)) as tf:
                arr = tf.pages[0].asarray()
                tr = tf.pages[0].tags["ModelTransformationTag"].value      # x0, y0 = top-left corner
            px, x0, y0 = tr[0], tr[3], tr[7]
            h, w = arr.shape
            xs = x0 + (np.arange(w) + 0.5) * px                            # pixel-centre coordinates
            ys = y0 - (np.arange(h) + 0.5) * px
            ci = np.floor((xs - (cx - half)) / C.CELL_M).astype(int)       # cell column, grid anchored on the site
            cj = np.floor((ys - (cy - half)) / C.CELL_M).astype(int)
            n_side = int(2 * half / C.CELL_M) + 1
            cell = cj[:, None] * n_side + ci[None, :]
            key = cell.astype(np.int64) * 256 + arr.astype(np.int64)
            uniq, cnt = np.unique(key, return_counts=True)
            for k, c in zip(uniq, cnt):
                cid, cls = divmod(int(k), 256)
                if cls == 0:                                               # NLCD 0 = no data
                    continue
                j, i = divmod(cid, n_side)
                mx, my = cx - half + (i + 0.5) * C.CELL_M, cy - half + (j + 0.5) * C.CELL_M
                out_rows.append((name, year, cid, mx, my, cls, int(c)))
            print(f"  NLCD {name} {year}: {h}x{w} pixels")
    df = pd.DataFrame(out_rows, columns=["site", "year", "cell_id", "cell_x_m", "cell_y_m", "nlcd_class", "pixels"])
    lon_c, lat_c = to_lonlat.transform(df["cell_x_m"].values, df["cell_y_m"].values)
    df.insert(5, "cell_lon", lon_c.round(5)); df.insert(6, "cell_lat", lat_c.round(5))
    df.to_csv(C.F_NLCD_RAW, index=False)
    print(f"Wrote {len(df):,} rows -> {C.F_NLCD_RAW}")


def gather_effort() -> None:
    """
    Reporting effort: how many eBird records of ANY species GBIF holds inside each site's search box
    per year. One request per site with a year facet.
      https://api.gbif.org/v1/occurrence/search?datasetKey=<eBird>&decimalLatitude=..&decimalLongitude=..
          &hasCoordinate=true&year=2010,2024&facet=year&facetLimit=30&limit=0
    """
    import math
    dataset = resolve_dataset_key()
    rows = []
    for name, _state, lat, lon, _role in C.SITES:
        dlat = C.SITE_RADIUS_KM / 111.0
        dlon = C.SITE_RADIUS_KM / (111.0 * math.cos(math.radians(lat)))
        js = get_json(f"{C.GBIF_API}/occurrence/search", {
            "datasetKey": dataset, "hasCoordinate": "true", "year": f"{C.YEARS[0]},{C.YEARS[-1]}",
            "decimalLatitude": f"{lat - dlat:.4f},{lat + dlat:.4f}",
            "decimalLongitude": f"{lon - dlon:.4f},{lon + dlon:.4f}",
            "facet": "year", "facetLimit": 40, "limit": 0})
        for f in js["facets"][0]["counts"]:
            rows.append((name, int(f["name"]), int(f["count"])))
        print(f"  effort {name}: {sum(r[2] for r in rows if r[0] == name):,} records")
    pd.DataFrame(rows, columns=["site", "year", "all_species_records"]).sort_values(["site", "year"]).to_csv(
        C.F_EFFORT_RAW, index=False)
    print(f"Wrote effort -> {C.F_EFFORT_RAW}")


# ------------------------------------------------------------------ 6. created dataset
def write_sites() -> None:
    df = pd.DataFrame(C.SITES, columns=["site", "state", "lat", "lon", "role"])
    df.to_csv(C.F_SITES, index=False)
    print(f"Wrote {len(df)} reference sites -> {C.F_SITES}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["gbif", "gbif_sites", "ebird", "ncei", "openmeteo", "landcover", "effort", "sites"],
                    help="gbif = random sample across North America, gbif_sites = every record near each "
                         "reference site, ncei = NOAA station weather")
    ap.add_argument("--quick", action="store_true", help="2020-2024, 300 records per cell")
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="GBIF step only: restrict to a range of years (lets several runs share the work)")
    ap.add_argument("--sites", nargs="+", help="gbif_sites step only: names (or parts of names) of the sites to pull")
    ap.add_argument("--countries", nargs="+", help="GBIF step only: subset of countries, e.g. --countries CA")
    args = ap.parse_args()
    steps = {"sites": write_sites,
             "ncei": gather_ncei,
             "openmeteo": gather_openmeteo,
             "landcover": gather_landcover,
             "effort": gather_effort,
             "gbif_sites": lambda: gather_gbif_sites(args.sites),
             "ebird": gather_ebird_api,
             "gbif": lambda: gather_gbif(args.quick, args.countries, tuple(args.years) if args.years else None)}
    for name, fn in steps.items():
        if args.only in (None, name):
            print(f"\n=== {name} ===")
            fn()
