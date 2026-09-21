"""
Shared settings for the sandhill crane project (Module 1).

Every script imports from here so that years, regions, file locations and
reference sites are defined in exactly one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
FIGS = ROOT / "images" / "eda"
for _p in (RAW, CLEAN, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- GBIF / eBird
GBIF_API = "https://api.gbif.org/v1"
SPECIES_NAME = "Antigone canadensis"          # GBIF backbone name for Sandhill Crane
# GBIF dataset that republishes eBird sightings ("EOD - eBird Observation Dataset").
# 01_gather_data.py verifies this key against the GBIF dataset endpoint before use.
EBIRD_DATASET_KEY = "4fa7b334-ce0d-4e88-aaae-2e0c138d049e"

YEARS = list(range(2010, 2025))               # full calendar years; GBIF's eBird copy has no 2025 records yet
COUNTRIES = ["US", "CA"]                      # add "MX" for the wintering range
RECORDS_PER_CELL = 600                        # cap per (country, year, month)
SLICE = 300                                   # GBIF max page size
RANDOM_SEED = 5612

# eBird API 2.0 (key required: https://ebird.org/api/keygen)
EBIRD_API = "https://api.ebird.org/v2"
EBIRD_SPECIES_CODE = "sancra"                 # Sandhill Crane
EBIRD_REGIONS = ["US-CO", "US-NE", "US-NM", "US-KS", "US-TX", "US-ID", "US-WY"]
EBIRD_HOTSPOT_REGIONS = ["US-CO", "US-NE", "US-NM", "US-KS", "US-TX", "US-ID", "US-WY"]

# ---------------------------------------------------------------- weather: NOAA NCEI
# GHCN-Daily station observations. Station lists are public files; observations come from the
# NCEI Access Data Service (no key).
NCEI_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
NCEI_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
NCEI_DATA_API = "https://www.ncei.noaa.gov/access/services/data/v1"
NCEI_DATATYPES = ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "SNWD", "AWND"]
NCEI_CORE = ["TMAX", "TMIN", "PRCP"]             # a station must report these across the study years
NCEI_SEARCH_KM = (60, 100)                       # try 60 km first, then widen to 100 km

WEATHER_START = f"{YEARS[0]}-01-01"
WEATHER_END = f"{YEARS[-1]}-12-31"
SITE_RADIUS_KM = 60                           # observations this close count toward a site

# ---------------------------------------------------------------- weather: Open-Meteo (supplement)
# Modelled (reanalysis) weather on a grid. Used for what NOAA stations do not report at every site:
# snowfall, snow depth and wind. Also gives a check on the station temperatures.
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_DAILY_VARS = ["temperature_2m_mean", "snowfall_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
                 "wind_direction_10m_dominant"]
OM_HOURLY_VARS = ["snow_depth"]                      # hourly only; summarized to a daily maximum

# ---------------------------------------------------------------- land cover: USGS NLCD (via MRLC WCS)
# 30 m national land cover, served by the MRLC Web Coverage Service. Each request returns only a small
# GeoTIFF clip around one site, so no national raster is ever downloaded.
NLCD_WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
NLCD_YEARS = [2008, 2011, 2013, 2016, 2019, 2021]      # epochs published as separate NLCD products
NLCD_COVERAGE = "mrlc_download__NLCD_{year}_Land_Cover_L48"
NLCD_CLIP_KM = 62                                       # half-width of the clip around each site (covers the 60 km radius)
CELL_M = 5000                                           # side of the analysis grid cells (5 km)
# NLCD class codes -> broad groups used in the analysis
NLCD_GROUPS = {
    "developed": [21, 22, 23, 24], "cropland": [82], "pasture_hay": [81],
    "grass_shrub": [52, 71], "forest": [41, 42, 43], "wetland": [90, 95],
    "open_water": [11], "barren": [31],
}

# ---------------------------------------------------------------- files
F_GBIF_RAW = RAW / "gbif_ebird_sandhill_raw.csv"          # random sample across North America
F_GBIF_SITES_RAW = RAW / "gbif_ebird_sites_raw.csv"       # every record near a reference site
F_EBIRD_API_RAW = RAW / "ebird_api_recent_raw.csv"
F_HOTSPOTS_RAW = RAW / "ebird_hotspots_raw.csv"
F_NCEI_RAW = RAW / "ncei_daily_weather_raw.csv"
F_NCEI_STATIONS = RAW / "ncei_stations_used.csv"
NCEI_CACHE = RAW / ".ncei_cache"
OM_CACHE = RAW / ".openmeteo_cache"
F_SITES = RAW / "sites_reference.csv"
F_OPENMETEO_RAW = RAW / "openmeteo_daily_weather_raw.csv"
F_NLCD_RAW = RAW / "nlcd_cell_class_pixels_raw.csv"       # pixel counts per 5 km cell, year and NLCD class
F_EFFORT_RAW = RAW / "gbif_ebird_effort_raw.csv"           # all-species eBird record counts per site and year

F_OBS_CLEAN = CLEAN / "observations_clean.csv"
F_SITE_OBS_CLEAN = CLEAN / "site_observations_clean.csv"
F_SITE_DAILY_CLEAN = CLEAN / "site_daily_clean.csv"
F_EBIRD_API_CLEAN = CLEAN / "ebird_api_recent_clean.csv"
F_HOTSPOTS_CLEAN = CLEAN / "ebird_hotspots_clean.csv"
F_LANDCOVER_CLEAN = CLEAN / "landcover_cell_year_clean.csv"
F_SITE_LANDCOVER_CLEAN = CLEAN / "landcover_site_year_clean.csv"
F_EFFORT_CLEAN = CLEAN / "site_year_effort_clean.csv"
F_CELL_CHANGE_CLEAN = CLEAN / "cell_change_clean.csv"

# ---------------------------------------------------------------- reference sites
# Created dataset. Reference-site coordinates are approximate points used to associate 
# nearby observations and environmental data; they do not represent the full boundary of each refuge.and are used
# only to pull weather and to group nearby observations. First five sites are named
# in the sources cited on the Introduction page; the last two are additional
# well-known crane areas and should be double-checked before being cited.
SITES = [
    # name,                          state, lat,    lon,      role
    ("Platte River - Kearney",        "NE", 40.700,  -99.080, "spring staging (Central Platte)"),
    ("Platte River - Rowe Sanctuary", "NE", 40.700,  -98.800, "spring staging (Central Platte)"),
    ("Monte Vista NWR",               "CO", 37.470, -106.150, "stopover (San Luis Valley)"),
    ("Bosque del Apache NWR",         "NM", 33.800, -106.890, "wintering"),
    ("Grays Lake NWR",                "ID", 43.300, -111.440, "breeding"),
    ("Quivira NWR",                   "KS", 38.150,  -98.500, "stopover (added; verify)"),
    ("Muleshoe NWR",                  "TX", 34.190, -102.720, "wintering (added; verify)"),
]

# ---------------------------------------------------------------- season labels
def season_of(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall"}[month]

def phase_of(month: int) -> str:
    """Three-way label used for colouring: northbound, southbound, other."""
    if month in (2, 3, 4):
        return "spring migration (Feb-Apr)"
    if month in (9, 10, 11):
        return "fall migration (Sep-Nov)"
    return "breeding / winter (May-Aug, Dec-Jan)"
