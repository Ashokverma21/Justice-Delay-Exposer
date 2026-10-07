# Judicial Delay Exposer

**Predicting and explaining how long Indian lower-court cases take, across 15 states.**

A machine-learning model trained on ~905,000 cases from the [Development Data Lab (DDL) Judicial Data](https://www.devdatalab.org/judicial-data) (2010–2018), with a local web UI, a full reproducible pipeline and a findings deck.

![Predict a case](docs/ui_predict.png)

## Highlights

| | |
|---|---|
| Data | 905,261 cases, 15 states, filed/decided 2010–2018 |
| Target | Time from filing to decision (days), modelled as `log1p(days)` |
| Model | Gradient boosting (scikit-learn `HistGradientBoostingRegressor`) with out-of-fold target encoding |
| Held-out accuracy | R² (log scale) **0.71** from filing-time information, **0.75** when the hearing purpose is also known |
| "Will it take more than a year?" | **82.3%** correct vs 62.9% for always guessing the majority (AUC 0.91) |
| Typical error | MAE ≈ 217 days on the held-out set; a single case can be off by 2–3x (the UI shows an 80% range) |

### Accuracy ladder (held-out 20%)

| Features | R² (log) |
|---|---|
| Court, case type, filing date, gender fields, judge-vacancy features | 0.666 |
| + act / section information | 0.709 |
| + purpose of hearing (no act) | 0.725 |
| + purpose of hearing + act / section | 0.754 |

### What the data says

- **State matters a lot.** Median days to decision range from **459 (Odisha)** to **80 (Karnataka)** across the 15 states.
- **Case type matters even more.** Median: bail 10 days, petty/traffic 35, criminal 136, motor accident 283, civil 422, negotiable instruments 443.
- **Judge vacancy.** Cases whose court had no sitting judge during the first 180 days after filing: median **148 days** vs **106** otherwise (judge roster matched for ~34% of cases, so treat as indicative).
- **Lok Adalat.** Cases listed for Lok Adalat: median **96 days** vs **190** for the rest, but the effect varies a lot by state.

![Findings tab](docs/ui_findings.png)

## Honest notes on accuracy

- The listing dates (`date_last_list`, `date_next_list`) are **deliberately not used**: they leak the outcome (R² ≈ 0.98) and are unavailable when a case is filed.
- Hearing purpose is only known after the first listing, so the 0.75 model is "early in the case", not "at filing".
- Cases still pending at the end of 2018 are not in the data (right-censoring), so very long durations are under-represented.
- The model has no knowledge of case facts, adjournments or judge workload on a given day.
- Valid only for the 15 states and 2010–2018. This is a research / hackathon project, not legal advice.

## Quick start (use the trained model)

```bash
git clone https://github.com/<your-username>/judicial-delay-exposer.git
cd judicial-delay-exposer
pip install -r requirements.txt
```

Download the two model files from the **Releases** page (or from the `models/` folder if included) and put them in `models/`:

```
models/model_filing_time.pkl.gz     # filing-time model
models/model_with_purpose.pkl.gz    # + hearing purpose
```

Run the UI:

```bash
python app.py
# open http://127.0.0.1:5000
```

Tabs: **Predict a case**, **Batch (CSV)** (upload many cases, download `predictions.csv`), **Findings**, **About / limits**.

> The pickles were built with `scikit-learn==1.9.1`, `pandas==3.0.5`, `numpy==2.5.3` (Python 3.12/3.13 recommended). Use the pinned versions, or retrain (below).

### Use the model from Python

```python
import gzip, pickle, pandas as pd
from delay_model import DelayModel          # required to unpickle

m = pickle.load(gzip.open("models/model_filing_time.pkl.gz", "rb"))
cases = pd.read_csv("my_cases.csv")         # same columns as the sample files
days = m.predict(cases)                     # predicted days; add acts=acts_df if available
```

Required columns: `ddl_case_id, year, state_code, dist_code, court_no, judge_position, female_defendant, female_petitioner, female_adv_def, female_adv_pet, type_name, date_of_filing` (plus `purpose_name` for the with-purpose model). Optional acts table: `ddl_case_id, act, section, bailable_ipc, number_sections_ipc, criminal`.

Command-line example: `python predict_example.py --model models/model_filing_time.pkl.gz --cases sample_cases.csv.gz --acts sample_acts.csv.gz --n 20`

## Reproduce from raw data

The DDL raw files are **not** included in this repository; download them from DDL and see their terms of use.

```bash
# 1. Streaming, low-RAM sample: 15 states x ~70k cases
python sample_states_final.py --data-dir D:\ddl --out-dir D:\sample --target 70000

# 2. Keep only the sampled cases from the (multi-GB) act/section file
python extract_acts.py --acts "path/to/acts_sections.csv" --sample D:\sample\sample_cases.csv.gz --out D:\sample\sample_acts.csv.gz

# 3. Train, evaluate, write model + results
python train_final.py --cases D:\sample\sample_cases.csv.gz --acts D:\sample\sample_acts.csv.gz --ddl D:\ddl --out results

# 4. Refresh the UI dropdowns / error bands (optional)
python make_lookup.py --cases D:\sample\sample_cases.csv.gz --acts D:\sample\sample_acts.csv.gz --ddl D:\ddl --pred results\predictions_test.csv.gz --out lookup.json
```

`--ddl` must contain the DDL key files (`type_name_key.csv`, `purpose_name_key.csv`, `disp_name_key.csv`, `act_key.csv`) and `judges_clean.csv`. If you run low on RAM, add `--limit 400000` to training.

## Repository layout

```
app.py                    Flask web UI + API
page.html                 UI front-end (no external dependencies)
delay_model.py            feature engineering, target encoding, DelayModel class
sample_states_final.py    chunked state-wise sampler
extract_acts.py           act/section extractor
train_final.py            training + evaluation + analysis
make_lookup.py            builds lookup.json for the UI
predict_example.py        CLI prediction example
lookup.json, results.json UI dropdown values and aggregate results
ppt/                      findings deck (+ generator script)
models/                   put the .pkl.gz model files here
docs/                     screenshots
```

## How it works (short)

1. **Labels:** DDL key tables are year-specific, so type / purpose / disposition codes are mapped per year and grouped into readable buckets (bail, civil, criminal, motor accident, Lok Adalat, …).
2. **Judge vacancy:** `judges_clean.csv` tenure rosters are turned into per-court gap calendars; a case gets a "vacancy in first 180 days after filing" feature.
3. **Target encoding:** court, district, case type, act and combinations are encoded out-of-fold (5 folds, smoothed) to avoid leakage.
4. **Model:** one gradient-boosting model per mode (`filing_time`, `with_purpose`), early-stopped, 80/20 random split, seed 42.

## Acknowledgements

Data: Development Data Lab, *Judicial Data* (eCourts). Built for a hackathon.
