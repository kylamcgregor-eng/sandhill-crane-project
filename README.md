# Sandhill Crane Migration Over Time (CSCI 5612 project, Part 1)

Static website (plain HTML/CSS) plus the Python code that gathers, cleans and explores the data.

## Run the pipeline

```bash
cd code
pip install -r requirements.txt

# 1. free eBird API key: https://ebird.org/api/keygen  (optional but recommended)
export EBIRD_API_KEY="your-key"          # Windows PowerShell: $env:EBIRD_API_KEY="your-key"

python 00_intro_schematic.py             # Introduction image (already included)
python 01_gather_data.py --quick         # test run: 2020-2024 sightings. Drop --quick for 2010-2024 (the GBIF copy of eBird has no 2025 records yet).
                                         # (also downloads NOAA station lists, ~40 MB, on first run)
python 02_clean_data.py                  # writes data/clean/*.csv and cleaning_log.json
python 03_eda_figures.py                 # writes images/eda/*.png (22 figures + 4 previews)
```

`01_gather_data.py` is resumable (progress is cached in `data/raw/.gbif_cells.jsonl` and `.gbif_site_cells.jsonl`) and can run one
source at a time: `--only gbif | gbif_sites | ebird | ncei | openmeteo | landcover | effort | sites` (`gbif` = continent-wide sample, `gbif_sites` = every record near each
reference site, `ncei` = NOAA station weather, `openmeteo` = modelled snowfall, wind and snow depth, `landcover` = NLCD land cover around each site, `effort` = all-species eBird record counts used to adjust for birdwatching effort).
The Open-Meteo step caches each finished site in `data/raw/.openmeteo_cache/`; its hourly snow-depth request is much larger than the daily one and the free service may refuse it once a daily allowance is used up. Re-run the step later and it resumes. If the connection is unreliable, `--countries` / `--years` / `--sites` split the GBIF pulls into pieces that can run side by side. Run `--only ebird` again every few weeks to
extend the eBird API time series (the API only looks back 30 days).

## Publish with GitHub Pages

1. Create a GitHub repository and push this folder to it (`data/`, `images/`, `code/` included).
2. Repository **Settings -> Pages -> Deploy from a branch -> main / (root)**.
3. The site URL (`https://<user>.github.io/<repo>/`) is what gets submitted.

Raw/clean CSV links on the site are relative paths, so they work once the files are committed.
Keep every file under 100 MB (GitHub's limit); lower `RECORDS_PER_CELL` in `code/config.py` if needed.

## Layout

| Path | Contents |
|------|----------|
| `index.html` ... `about.html` | One page per tab: Introduction, DataPrep_EDA, Clustering, PCA, NaiveBayes, DecTrees, SVMs, Regression, NN, Conclusions, About Me |
| `code/` | Python scripts (`config.py` holds all settings) |
| `data/raw`, `data/clean` | Raw API output (never edited) and cleaned files |
| `images/` | Introduction image; `images/eda/` holds figures |
