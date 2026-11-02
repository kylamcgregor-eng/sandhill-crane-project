# Sandhill Crane Migration Over Time (CSCI 5612 project, Part 3)

Static website (plain HTML/CSS) plus the Python code that gathers, cleans, explores, clusters,
reduces the dimensionality of, and classifies the data (Naive Bayes and Decision Trees).

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
python 04_clustering_pca.py              # writes data/clean/cluster_features.csv, clustering_pca_log.json,
                                         # images/clustering/*.png, images/pca/*.png (Part 2: k-means,
                                         # hierarchical clustering, and PCA on unlabeled numeric features)
python 05_concept_diagrams.py            # writes the made-up/illustrative Overview diagrams used on the
                                         # Clustering and PCA tabs (not run on project data)
python 06_naive_bayes.py                 # writes the shared labeled train/test split (data/clean/nb_dt_*.csv,
                                         # stratified by season, reused by 07) and fits Multinomial + Bernoulli
                                         # NB; writes images/naivebayes/*.png and naive_bayes_log.json
python 07_decision_trees.py              # loads the split written by 06, runs the max_depth sweep, fits three
                                         # trees (gini/entropy shallow + practical-depth gini), and reproduces a
                                         # real Gini/Entropy/Information-Gain worked example from the fitted root
                                         # split; writes images/dectrees/*.png and decision_trees_log.json
python 08_concept_diagrams2.py           # writes the made-up/illustrative Overview diagrams used on the
                                         # NaiveBayes and DecTrees tabs (not run on project data)
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
| `index.html` ... `about.html` | One page per tab: Introduction, DataPrep/EDA, Clustering, PCA, NaiveBayes, DecTrees, Regression, AdaBoost, GradientBoosting, XGBoost, Conclusions, About Me |
| `code/` | Python scripts (`config.py` holds all settings) |
| `data/raw`, `data/clean` | Raw API output (never edited) and cleaned files, including `cluster_features.csv` (unlabeled numeric data used by Clustering/PCA) and the labeled Part 3 split: `nb_dt_all_labeled.csv`, `nb_dt_sample.csv`, `nb_dt_train.csv`, `nb_dt_test.csv`, `naive_bayes_log.json`, `decision_trees_log.json` |
| `images/` | Introduction image; `images/eda/` holds EDA figures, `images/clustering/` and `images/pca/` hold the Part 2 figures, `images/naivebayes/` and `images/dectrees/` hold the Part 3 figures |

## Project parts

- **Part 1** - topic, data gathering/cleaning, DataPrep/EDA tab, site skeleton.
- **Part 2** - Clustering tab (k-means + hierarchical/cosine) and PCA tab, both run on
  `cluster_features.csv` (normalized latitude, longitude, day of year, year, and flock size - no
  season/site/label columns).
- **Part 3** (this version) - NaiveBayes and DecTrees tabs, both supervised and both run on the same
  disjoint, stratified train/test split of labeled data (label: `season`; features: `lat_norm`,
  `lon_norm`, `log10_count` - date-derived columns are excluded since `season` is defined from the
  month). Also added a fifth paragraph to the Introduction.
- **Part 4** (later) - Regression and AdaBoost tabs (due Fri Nov 20).
- **Part 5** (later) - GradientBoosting and XGBoost tabs, plus the finished site (due Fri Dec 4).

The site's tab bar already reflects the final structure: the old SVMs and NN tabs have been
replaced with Regression, AdaBoost, GradientBoosting, and XGBoost (`regression.html`,
`adaboost.html`, `gradientboosting.html`, `xgboost.html`), each set up as a placeholder until its
part is due.

Each part was built on its own git branch off `main` and merged in once finished, so earlier
submissions are never modified by later work.
