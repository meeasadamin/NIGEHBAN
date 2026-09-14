# AUDIT.md

This file has two parts:

1. **Pre-rebuild audit (verbatim).** The audit of v1 as delivered on 2026-09-13, reproduced unedited. It describes the v1 code, which is preserved in `archive/v1/`. Its "What I've changed so far" section describes the state at the time of writing, not the final state.
2. **Traceability index (added during the v2 rebuild).** Each finding gets an ID, its v1 file/line location, the resolution, and how the resolution is verified. Findings discovered during the rebuild are listed separately.

---

## Part 1 — Pre-rebuild audit (verbatim)

# Audit brief: Pakistan Multi-Hazard Risk Analyzer

## Summary

1. **The model file doesn't match the data.** `models/best_hazard_pipeline.pkl` was trained on a different dataset than `data/pakistan_districts.csv`. The dashboard's predictions don't correspond to the districts it shows. This is the most serious problem.
2. **Your reported scores don't agree with each other.** The README, the notebook and the model file each give different model types and F1 scores.
3. **The What-If sliders can mislead users.** Past the range the model was trained on, predictions stop changing and nothing warns the user.
4. **The engineering has a solid base.** The data generator reproduces the CSV exactly, the code has logging and error messages, Agg is set correctly, and you used macro-F1 with stratified folds. Most of the fixes are corrections, not a rebuild.

**Labels used below:** **[Ran it]** means I measured or reproduced it. **[Code]** means I found it by reading the code but haven't run it.

---

## Layer 1: Statistics and ML pipeline

### 1.1 Model and data mismatch (critical) [Ran it]
- **Evidence 1:** a fitted scaler stores the medians of its training data, and the shipped model's medians don't match your CSV:

  | Feature | Median inside model | Median in CSV |
  |---|---|---|
  | Annual rainfall | 236.5 | 349.9 |
  | NDVI | 0.28 | 0.43 |
  | Past flood events | 2 | 4 |

- **Evidence 2:** the model scores worse on its supposed training data (F1 0.59 / 0.53 / 0.69) than a fresh 5-fold cross-validation does (0.68 / 0.58 / 0.79). That is almost impossible for a correctly trained model. Refitting on the CSV gives 0.89 / 0.995 / 0.89 on the training data.
- **Evidence 3:** the `.pkl` was saved on Sep 11, after the CSV and the notebook run (Sep 10).
- **Impact:** every prediction and SHAP chart in the dashboard is unreliable. A symptom: for Karachi, the flood model's top driver is NDVI, which isn't in the flood formula at all.

### 1.2 Three sources of truth disagree [Ran it]

| Source | Models | Macro-F1 (flood / heat / seismic) |
|---|---|---|
| README and notebook output | CatBoost ×3 | 0.716 / 0.665 / 0.791 |
| Shipped `.pkl` (shown in the app footer) | RandomForest / CatBoost / RandomForest | 0.790 / 0.601 / 0.833 |

An interviewer who opens the `.pkl` will see this immediately.

### 1.3 Calibration: the probabilities are not calibrated [Ran it]
Calibration was never measured anywhere in the notebook (no Brier score, log loss or reliability chart). Using out-of-fold predictions:

| Hazard | Model says | Actually correct |
|---|---|---|
| Flood (RF) | 59% | 75% (underconfident) |
| Flood (RF) | 78% | 96% (underconfident) |
| Seismic (RF) | 59% | 75% (underconfident) |
| Heatwave (CatBoost) | 79% | 52% (overconfident) |
| Heatwave (CatBoost) | P(High) ≥ 50% | only 33% were actually High |

- **Causes:** `class_weight="balanced"` inflates minority-class probabilities. RandomForest averaging pulls probabilities toward the middle.
- **Sigmoid calibration (tested properly):** it improved flood and seismic log loss (0.677 → 0.645 and 0.588 → 0.565). For heatwave it lowered F1 (0.576 → 0.514). With only 18–23 "High" samples per hazard, calibration is fragile.
- **Recommendation:** report log loss, Brier score and a reliability chart. Rename "Model confidence" to "Model score" until the models are calibrated.

### 1.4 Class imbalance and metrics [Ran it + Code]
- **Good:** macro-F1, stratified folds, and class weights for RandomForest and CatBoost.
- **Unfair baseline:** XGBoost got no class weighting, so it lost the model comparison partly for that reason.
- **Missing metrics:** no ROC-AUC, log loss or "High"-class recall. Measured values:
  - ROC-AUC: 0.87 / 0.83 / 0.90
  - High recall: 0.74 / **0.39** / 0.85
  - **The heatwave model misses 61% of truly High districts**, the most important number for NDMA.
- **Ordinal labels:** Low < Medium < High is ordered, but it's treated as unordered. Calling a High district Low should count as worse than calling it Medium. Add quadratic weighted kappa.

### 1.5 Reported scores are inflated [Ran it]
The model family was chosen, and Optuna tuned, on the **same 5 folds** later used to report the score. That picks the luckiest of 30+ tries.
- **Evidence:** the same hyperparameters on 5 different fold shuffles give:
  - flood 0.677 ± 0.019
  - heatwave 0.584 ± 0.036
  - seismic 0.751 ± 0.025

  Seismic scores 0.790 on the original folds but 0.72–0.76 on the others.
- Your "0.702 → 0.716 after tuning" gain is within the noise.
- **Fix:** nested cross-validation, and report mean ± std.

### 1.6 Leakage audit of `generate_data.py`
- **No direct leakage [Ran it]:** `overall_risk_score` is built from the labels and is correctly left out of the model features.
- **Latent risk:** that column still sits in the CSV and the correlation heatmap, so one careless edit would add it as a feature.
- **Data quality problems:**
  - `historical_disasters` is exactly floods + earthquakes, so SHAP splits credit between them arbitrarily. Two rows even break that sum, because outlier injection adds earthquakes after it's computed. [Ran it]
  - Outlier rows aren't rounded (e.g. `7672.811233`), which reveals which rows are injected. [Ran it]
  - Physically impossible values [Ran it]:
    - Sialkot at 7,673 m and Gujrat at 7,134 m (both Punjab plains)
    - Faisalabad at 3 people/km²
    - Rawalpindi at latitude 29.6 (really about 33.6)
  - 44 of 150 districts are invented names ("Lahore Rural-2"). [Ran it]
  - Latitude and longitude are sampled in a rectangle, so about 6 points fall roughly outside Pakistan (rough bounding-box check). [Ran it]
  - Seismic risk is 40% driven by **elevation**, and Balochistan gets no seismic boost despite the Quetta (1935) and Awaran (2013) earthquakes. Real seismic hazard follows fault lines (Chaman, Main Boundary Thrust, Makran). An NDMA reviewer will spot this.
  - Labels are relative: the top ~15% are "High", using min-max scaling over the whole dataset. One extreme outlier compresses everyone else's scores.
  - The README says the models "recovered physically sensible relationships without being told". That's circular: the generator wrote those relationships.
  - Dead code: `_temp_proxy` is described as "used for heatwave label" but is never used.
  - All 3 final pipelines share **one** preprocessor object. It's harmless today but a latent bug.

---

## Layer 2: Code architecture [Code]

- **Monolith:** `app.py` mixes loading, ML, SHAP and UI in 480 lines.
  - The target list is hard-coded 4 times, although the bundle already stores it.
  - The 0/1/2 → Low/Medium/High mapping is repeated 3 times.
  - The notebook duplicates the SHAP handling.
- **Wrong type hint:** `render_sidebar` is declared to return a DataFrame but returns a 3-tuple.
- **Missing type hints and docstrings:**
  - No types on `build_shap_explainer`.
  - No return type on `compute_shap_for_prediction`.
  - No docstring on `main`.
  - The bundle is an untyped `dict[str, Any]`.
  - Docstrings are mostly prose, not Google style. `generate_data.py` is the exception and is well typed.
- **Comments that are false:**
  - "cached by content": it's cached by the path string, so an updated CSV isn't picked up until restart.
  - The `model_bundle_path` cache key "invalidates when the model changes": it doesn't, because the string stays the same.
  - "Agg is thread-safe": the backend may be, but pyplot's global state is not (see 3.1).
- **Dead code:** `RISK_COLORS` is never used, and `reset_sliders` is set but never read.
- **Bug:** `clean_feature_name` uses `.capitalize()`, which lowercases the rest of the string. "Khyber Pakhtunkhwa" becomes "Khyber pakhtunkhwa".
- **Memory:** your DataFrame is 22 KB, so converting to float32 would save about 15 KB and could slightly shift predictions. It's not worth doing. Use `category` dtype for **validation** (a fixed list of provinces), not for memory. I checked that the rows passed to the model keep correct dtypes. **Pass.** [Ran it]

---

## Layer 3: Streamlit concurrency and state

### 3.1 SHAP chart race between users (real bug) [Ran it]
- `matplotlib.use("Agg")` is set correctly, before pyplot is imported. **Pass.**
- **The race:** `shap.plots.waterfall` draws on pyplot's global "current figure" (`plt.gcf()`). When two users render at the same moment, one can draw on the other's figure.
  - **Reproduced:** 6 concurrent sessions × 100 renders → **6 of 600 charts came out blank or showed another session's chart.**
- **Side effects:**
  - Your `figsize=(10, 6)` is ignored, because SHAP resizes the figure itself.
  - `plt.close(fig)` isn't in a `finally` block, so figures leak on errors.
- **Best fix:** draw the waterfall with Plotly, which has no global state.

### 3.2 Cache mutability [Code]
- `@st.cache_data` returns a fresh copy on each call, and the code never mutates cached data. **Pass.**
- **Cached failures:** the loaders catch errors and return `None`, and Streamlit **caches that `None`**. A temporary failure, such as a file mid-upload during a redeploy, stays broken until the server restarts. Raise inside the cached function and catch outside it.
- **Stale model:** replacing the `.pkl` doesn't reload it until restart. Put the file's modification time in the cache key.
- **Unnecessary reruns:** every slider move reruns the whole page, including the map. Use `@st.fragment`.

---

## Layer 4: Edge cases and security

### 4.1 Silent extrapolation (most dangerous UX flaw) [Ran it]

| Slider | Slider maximum | Training maximum |
|---|---|---|
| Rainfall (mm) | 3,000 | 1,485 |
| River distance (km) | 200 | 121 |
| Population density (/km²) | 5,000 | 3,045 |
| Elevation (m) | 8,611 | 7,673 |

Rainfall at 1,490 mm and 3,000 mm gives **identical** predictions, with no warning. A responder testing "what if rainfall doubles?" gets false reassurance. Unrealistic combinations, like 5,000 m elevation in Karachi at 38 °C, aren't flagged either.

### 4.2 Graceful degradation
- **Good:** a clear message for missing files, and `try/except` around prediction and SHAP.
- **Gaps [Code, not run]:**
  - A CSV missing a column raises a `KeyError` in the sidebar before any `try` block.
  - An empty CSV crashes the district lookup.
  - A bundle without `cv_macro_f1` crashes the footer.
  - There's no top-level error handler, and Streamlit shows raw tracebacks to users unless configured not to.

### 4.3 Security
- No `eval`/`exec` and no user-supplied file paths, so no path traversal. **Pass.**
- **Pickle risk:** `joblib.load` runs code from the pickle. A tampered `.pkl` in the repo means code execution on the server. Fix: check a SHA-256 hash before loading.
- **Working-directory paths:** `Path("data/...")` is relative to where the app is launched, so starting it from another folder breaks it.
- Add a `.streamlit/config.toml` that hides error details and disables usage stats.

---

## Layer 5: UI/UX and accessibility

- **Gauge:** there is no gauge chart in the app today.
- **Contrast** [Ran it]:
  - Your amber `#F9A825` is 1.97:1 on white, which fails WCAG. It's in the unused `RISK_COLORS`, so the problem is latent.
  - SHAP's default red and blue pass for graphics (3:1) but fail for small text (4.5:1).
- **Colour-blindness:**
  - 🟢🟡🔴 are identical shapes, so red and green look alike to about 8% of men. The text label on the cards saves you.
  - The notebook charts use the red-to-green `RdYlGn_r` colormap, which fails.
- **New palette, tested** [Ran it]:
  - Low `#0077BB` ●, Medium `#CC6677` ◆, High `#CC3311` ▲.
  - Contrast is at least 3.6:1 on both light and dark themes.
  - Every pair stays distinct (ΔE ≥ 34.9) under protanopia, deuteranopia and tritanopia simulation.
  - An amber/vermillion pair I tried first collapsed to ΔE 2.4 for protanopes, which is why shapes are added too.
- **Information architecture problems:**
  - The progress bar shows confidence in the predicted class, so "95% sure it's **Low**" looks like a full, alarming bar.
  - The metric shows an ↑ arrow next to "confidence", implying an increase.
  - On mobile the sidebar is hidden, so the district selector can't be seen.
  - The first screen is a title and captions; the radar chart with only 3 axes is a weak chart; the one-dot map takes half the screen.
  - The SHAP chart is a static image that's unreadable on a phone.
  - It only explains the highest hazard, and its units silently differ: probability for RandomForest, log-odds for CatBoost.
  - There's no banner saying "you are viewing a simulated scenario", so edited values look like real risk.
  - The "not for operational use" disclaimer is small grey text at the bottom.

---

## Layer 6: MLOps and reproducibility

- **Dependency pins** [Ran it + Code]:
  - The ML libraries are pinned exactly. **Good.**
  - `streamlit>=1.38` is unpinned, but the code uses the newer `width='stretch'` API, so an older install could crash.
  - `plotly` is imported but **not listed**. It only works because CatBoost happens to depend on it.
  - `folium` and `streamlit-folium` are listed but never used.
  - Training-only libraries (optuna, xgboost, seaborn) get installed on the server.
  - `jupyter`, needed by the README instructions, isn't listed.
  - There's no Python version pin (numpy 2.4 and pandas 3.0 need Python 3.11+).
- **No provenance:** the bundle doesn't record the data hash or library versions. That's exactly how the mismatch in 1.1 slipped through.
- **Reproducibility:** `generate_data.py` reproduces the CSV bit for bit. **Pass.** [Ran it]
- **Testing:** no tests, no CI, no linting.
- **Tests I'd add:**
  - Probabilities sum to 1.
  - SHAP values add up to the model output.
  - Invalid inputs are rejected.
  - A Streamlit AppTest smoke test.
  - A model–data consistency test, which would have caught 1.1.
- **CI:** a GitHub Actions workflow that installs, lints (ruff) and runs pytest.

---

## Layer 7: What-If Simulator

- **Today:** the sidebar sliders already recompute predictions, so a basic version exists.
- **Missing:**
  - baseline vs. scenario comparison
  - a gauge
  - a choice of which hazard to explain
  - an extrapolation guard
  - a "simulated scenario" banner
  - mobile layout
  - partial reruns
- **Planned layout:**
  1. Risk cards (● ◆ ▲) that show the change from baseline.
  2. A "SIMULATED SCENARIO" banner.
  3. Sliders limited to the training range.
  4. A hazard picker, a Plotly gauge for P(High) with a ▲/▼ delta vs. baseline, and a Plotly SHAP waterfall.
  5. A plain-language list of the top drivers.
  6. Warnings for unusual combinations.

---

## What I've changed so far
- **Backup:** your originals are in `..\pakistan-multi-hazard-risk-analyzer_pre-audit-backup`.
- **New file [risk_engine.py](risk_engine.py):** the core logic, separated from the UI. I tested it against your real model:
  - It automatically flags the model/data mismatch.
  - It rejects invalid scenarios.
  - It merges province one-hot columns back into "Province = Sindh".
  - It labels SHAP units per model.
  - It detects unusual combinations; Karachi at 51 °C with NDVI 0.9 was flagged.
- **Not changed yet:** `app.py`, `requirements.txt`, the notebook and the generator. Nothing uses the new engine yet.
- **Running:** your original app is still at **http://localhost:8502**.

## Suggested fix order
- **P0, before submitting:**
  1. Retrain so the model, data and README agree.
  2. Correct the README's numbers and its "all CatBoost" claim.
  3. Fix `requirements.txt`: add plotly, pin streamlit, remove folium.
  4. Limit sliders to the training range.
  5. Replace the pyplot SHAP chart with Plotly.
- **P1:**
  1. Nested CV and a calibration report.
  2. Fix the cached-`None` bug.
  3. Add an error boundary.
  4. Mobile and accessibility layout.
  5. Tests and CI.
- **P2:**
  1. Generator realism: real district coordinates and a fault-based seismic proxy.
  2. Move training from the notebook into `train.py`.

## Decisions for your next prompt
1. **Retrain the model?** It will change your README scores, most likely downward, but the new scores will be honest.
2. **Data generator:** keep the data and fix only the integrity bugs, or make it realistic (real coordinates, fault-based seismic)? Making it realistic changes all results.
3. **App:** a full rewrite with the new What-If layout, or small patches to the current layout?
4. **CI Python version:** 3.12, matching your README badge?
5. **Report:** save this as `AUDIT.md` in the repo?

For example: *"Retrain, keep data but fix integrity bugs, full app rewrite, Python 3.12, add AUDIT.md."*

---

## Part 2 — Traceability index (added during the v2 rebuild)

v1 locations refer to the archived copies: `archive/v1/app_v1.py`, `archive/v1/generate_data_v1.py`, `archive/v1/requirements_v1.txt`, `archive/v1/README_v1.md`, and `analysis.ipynb` (cell ids; the notebook is kept as the v1 record). Priority follows "Suggested fix order" above. Status: **Resolved**, **Partial** (improved, with a stated remaining gap), **Not resolved** (with reason).

### Layer 1 — Statistics and ML pipeline

| ID | Pri | Finding | v1 location | Status | Resolution (v2) | Verified by |
|---|---|---|---|---|---|---|
| F1.1 | P0 | Model trained on different data than the CSV | `models/best_hazard_pipeline.pkl` (now `archive/v1/best_hazard_pipeline_v1.pkl`); `analysis.ipynb` cell `9e032089` | Resolved | `train.py` retrains from the committed CSV and records `data_sha256`; `risk_engine.audit_provenance` checks the hash, the RobustScaler training medians, and library versions at every app start | `tests/test_risk_engine.py::test_committed_model_matches_committed_data`, `::test_data_mismatch_is_detected` |
| F1.2 | P0 | README, notebook, and bundle disagree | `README_v1.md:100-108` "Model Performance" table; `app_v1.py:469-475` footer | Resolved | README metrics are copied from `models/metrics.json`; the app's Model card reads the same metadata from the bundle; notebook marked archived | Manual: README table vs `models/metrics.json` |
| F1.3 | P1 | Probabilities uncalibrated and never measured; "confidence" wording | notebook cells `d79deaa7`, `fb1660d8`; `app_v1.py:285-296` | Partial | Temperature scaling (`train.py::calibrate`), fixed in advance; Brier, log loss, ECE reported before/after on outer folds; UI says "calibrated probability". **Gap:** gains are modest and flood ECE got worse (see README) | `tests/test_risk_engine.py::test_temperature_calibration_never_changes_the_predicted_class`, `::test_bundle_records_brier_and_log_loss` |
| F1.4a | P1 | XGBoost compared without class weighting | notebook cell `09f6dc2c` `get_baseline_models` | Resolved (by removal) | Candidates are class-weighted RandomForest and CatBoost only; XGBoost dropped (sample-weight routing through calibration not implemented) | `train.py::suggest_classifier` |
| F1.4b | P1 | Missing ROC-AUC, High recall; ordinal labels | notebook cell `fb1660d8` | Partial | Balanced accuracy, ROC-AUC, High recall/precision, quadratic weighted kappa reported. **Gap:** no ordinal model or cost-sensitive decision threshold | `train.py::score_probabilities`, `tests/test_train.py` |
| F1.5 | P1 | Selection and scoring on the same folds | notebook cells `d79deaa7`, `df1d4443` | Resolved | Nested CV (5 outer × 3 inner, Optuna inside outer training folds); mean ± std reported. `scripts/evaluate_v1_protocol.py` quantifies the v1 optimism on v1 data | `archive/v1/v1_nested_cv_metrics.json` |
| F1.6-a | P2 | Random coordinates in a rectangle | `generate_data_v1.py:204-213` | Resolved | Real GeoNames district points (`scripts/build_reference_data.py`) | `tests/test_generate_data.py::test_districts_are_real_geonames_points` |
| F1.6-b | P2 | 44 invented "Rural-N" districts | `generate_data_v1.py:159-164` | Resolved | 150 real GeoNames ADM2 districts; exclusions logged in `data/reference/SOURCES.md` | `::test_no_invented_district_names` |
| F1.6-c | P2 | Physically impossible values | `generate_data_v1.py:214-217, 270-272` | Resolved | Real DEM elevation; bounded synthesis; `validate_physical_bounds` raises on violation | `::test_every_value_is_within_physical_bounds`, `::test_bounds_validation_rejects_impossible_values` |
| F1.6-d | P2 | Seismic label from elevation, not faults | `generate_data_v1.py:309-318` | Resolved | Distances to real GEM active faults and the Makran subduction zone drive the seismic score and simulated earthquake counts | `::test_seismic_geometry_is_physically_ordered`, `::test_chaman_segment_identification_holds` |
| F1.6-e | P2 | `historical_disasters` collinear; identity broken | `generate_data_v1.py:230-232, 272` | Resolved | Column removed; anomalies applied before dependent features | `::test_label_derived_and_collinear_columns_are_gone` |
| F1.6-f | P2 | Unrounded outliers reveal injected rows | `generate_data_v1.py:264-272` | Resolved | Single rounding pass after all synthesis | `::test_published_values_are_rounded_uniformly` |
| F1.6-g | P1 | Label-derived `overall_risk_score` in CSV | `generate_data_v1.py:333-341` | Resolved | Column removed; bundles using label-derived or identifier columns are rejected at load | `::test_label_derived_and_collinear_columns_are_gone`, `tests/test_risk_engine.py::test_leaky_bundle_is_rejected` |
| F1.6-h | P2 | Min-max label scaling squashed by outliers | `generate_data_v1.py:287-288` | Partial | Fixed physical reference ranges. **Gap:** labels remain relative quantile ranks (documented) | `generate_data.py::_derive_risk_labels` |
| F1.6-i | P2 | Dead `_temp_proxy` with false comment | `generate_data_v1.py:218` | Resolved | Removed | Code review |
| F1.6-j | P2 | "Circular" README claim; `historical_*` names imply real records | `README_v1.md:110`; `generate_data_v1.py:224-229` | Resolved | Claim removed from README; columns renamed `simulated_*`; `data/DATA_DICTIONARY.md` marks every column REAL or SYNTHETIC | `::test_label_derived_and_collinear_columns_are_gone` |
| F1.6-k | P2 | Shared preprocessor object across pipelines | notebook cells `f0dcd30f`, `fb1660d8` | Resolved | `train.py::make_pipeline` builds a new preprocessor per pipeline | `tests/test_train.py::test_each_pipeline_gets_its_own_preprocessor` |

### Layer 2 — Code architecture

| ID | Pri | Finding | v1 location | Status | Resolution (v2) | Verified by |
|---|---|---|---|---|---|---|
| F2.1 | P1 | Monolith, duplicated constants, `predict()` shape workaround | `app_v1.py` (all); `:120-127`, `:282`, `:318`, `:322`, `:460-462` | Resolved | `risk_engine.py` (data, models, SHAP, exports), `ui_theme.py` (design tokens), `app.py` (layout only); targets/labels defined once; `predict_proba` only | Test suite imports the engine without Streamlit |
| F2.2 | P1 | Wrong/missing type hints and docstrings | `app_v1.py:103`, `:152-154`, `:225/276`, `:400` | Resolved | All public functions typed with Google-style docstrings; bundle is a frozen `ModelBundle` dataclass | `ruff check` (CI) |
| F2.3 | P1 | False comments (cache by content, cache-key invalidation, Agg thread-safe) | `app_v1.py:20-24`, `:89`, `:108-110` | Resolved | Comments rewritten; cache keys include file mtime/size | `app.py::get_bundle`, `::get_districts` |
| F2.4 | P2 | Dead code (`RISK_COLORS`, `reset_sliders`) | `app_v1.py:40`, `:239-241`, `:425-426` | Resolved | Removed; palette lives in `ui_theme.py` and is used | Code review |
| F2.5 | P1 | `clean_feature_name` lowercases provinces | `app_v1.py:209-218` | Resolved | Explicit `FEATURE_LABELS`; one-hot columns folded back to "Province = Sindh" | `tests/test_risk_engine.py::test_one_hot_columns_are_folded_back_into_source_features` |
| F2.6 | — | Memory / dtypes | `app_v1.py:91` | Pass (no change needed) | `category` dtypes used for closed-vocabulary validation, not memory | `risk_engine.CSV_DTYPES` |

### Layer 3 — Streamlit concurrency and state

| ID | Pri | Finding | v1 location | Status | Resolution (v2) | Verified by |
|---|---|---|---|---|---|---|
| F3.1 | P0 | pyplot race between sessions; figure leak | `app_v1.py:374-378` | Resolved | Plotly waterfall; no matplotlib in serving code | `tests/test_no_pyplot.py` (AST scan), `tests/test_app_smoke.py::test_no_matplotlib_figure_is_ever_created` |
| F3.2a | P1 | Cached `None` on load failure | `app_v1.py:68-99` | Resolved | Cached loaders raise; exceptions are not cached | `tests/test_app_smoke.py::test_missing_model_fails_gracefully` |
| F3.2b | P1 | Stale model after file replacement | `app_v1.py:68-85` | Resolved | Cache key includes mtime and size | `app.py::get_bundle` |
| F3.2c | P1 | Shared cached objects mutable | `app_v1.py:68` | Resolved | Frozen dataclasses, `MappingProxyType`, read-only arrays | `tests/test_risk_engine.py::test_loaded_arrays_are_read_only` |
| F3.3 | P2 | Whole-page reruns on every slider move | `app_v1.py:251-262` | Not resolved | The brief places sliders in the sidebar, and `st.fragment` cannot write to the sidebar. Reruns are bounded: cached loaders, 3 SHAP calls | Design decision |

### Layer 4 — Edge cases and security

| ID | Pri | Finding | v1 location | Status | Resolution (v2) | Verified by |
|---|---|---|---|---|---|---|
| F4.1 | P0 | Silent extrapolation past training range; odd combinations unflagged | `app_v1.py:53-61` | Resolved | Sliders bounded to training min/max; engine flags out-of-range values (API use) and novel combinations (nearest-neighbour distance above the 95th percentile) | `tests/test_app_smoke.py::test_sliders_are_bounded_to_the_training_range`, `tests/test_risk_engine.py::test_applicability_domain_flags_extrapolation` |
| F4.2 | P1 | Tracebacks on schema errors; no error boundary | `app_v1.py:225-276`, `:471-473` | Resolved | Schema validation raises user-safe `RiskEngineError`s; top-level boundary with incident id; `client.showErrorDetails = "none"` | `tests/test_app_smoke.py::test_missing_model_fails_gracefully`, `tests/test_risk_engine.py::test_invalid_scenarios_are_rejected` |
| F4.3a | P1 | Unverified pickle load | `app_v1.py:76` | Resolved | SHA-256 verified before `joblib.load` (env var `NDMA_MODEL_SHA256` or manifest) | `tests/test_risk_engine.py::test_tampered_model_is_refused_before_unpickling` |
| F4.3b | P1 | Working-directory-relative paths | `app_v1.py:37-38` | Resolved | Paths resolved from `risk_engine.py`'s location (`PROJECT_ROOT`), overridable by environment variable | `risk_engine.default_data_path`, `::default_model_path` (code review) |
| F4.3c | P1 | No Streamlit hardening config | — | Resolved | `.streamlit/config.toml` | File review |

### Layer 5 — UI/UX and accessibility

| ID | Pri | Finding | v1 location | Status | Resolution (v2) | Verified by |
|---|---|---|---|---|---|---|
| F5.1 | — | No gauge chart | — | Not resolved | Not in the rebuild brief; KPI cards show P(High) and change vs unmodified inputs | — |
| F5.2 | P1 | Contrast unverified; failing amber | `app_v1.py:40` | Resolved | Every rendered colour pair checked against the composited surfaces actually drawn; Medium `#CC6677` (3.11:1) restricted to 28px bold (large text) and graphics | `tests/test_accessibility.py`; computed 28px/700 measured in Edge by `scripts/check_layout.py` |
| F5.3 | P1 | Colour-only cues; red/green | `app_v1.py:41`; notebook cells `4ea2d0fe`, `b86b44b5` | Partial | Shapes ● ◆ ▲, +/− signs, text labels, CVD-safe palette in the app; v1 EDA images moved to `archive/v1/images/`. **Gap:** the archived notebook and its images still use `RdYlGn_r` (kept unchanged as the v1 record) | `tests/test_accessibility.py::test_levels_never_rely_on_colour_alone` |
| F5.4 | P1 | Information architecture (confidence bar, arrow, hidden selector, radar, static SHAP, single hazard, units, banner, disclaimer) | `app_v1.py:279-393`, `:468-475` | Partial | Persistent banner at top; calibrated P(High) instead of "confidence"; radar removed; interactive per-hazard Plotly waterfalls with unit-labelled axes and a table view; insight sentence per hazard. **Gap:** on phones the district selector stays behind the sidebar toggle (brief requires sidebar placement), with an on-page hint | `tests/test_app_smoke.py`, `scripts/check_layout.py` |

### Layer 6 — MLOps and reproducibility

| ID | Pri | Finding | v1 location | Status | Resolution (v2) | Verified by |
|---|---|---|---|---|---|---|
| F6.1 | P0 | Unpinned Streamlit; plotly missing; unused folium; training libs on server | `requirements_v1.txt:3-18` | Resolved | `requirements.in` → uv-compiled full lock for Python 3.12 (`requirements.txt`); `requirements-dev.txt` for training/tests | Lock installed into a clean Python 3.12.13 venv; full test suite passed there |
| F6.2 | P1 | No provenance in bundle | notebook cell `9e032089` | Resolved | Bundle metadata: data hash, library versions, Python version, search config, nested-CV metrics; SHA-256 manifest | `tests/test_train.py::test_quick_training_run_produces_a_loadable_verified_bundle` |
| F6.3 | P1 | No tests or CI | — | Resolved | 72 tests; `.github/workflows/ci.yml` runs ruff, dataset reproducibility, tests, and a training smoke run | Local run of every CI step on Python 3.12 |

### Findings discovered during the rebuild

| ID | Finding | Status | Resolution | Verified by |
|---|---|---|---|---|
| R1 | `shap.TreeExplainer` (shap 0.52) intermittently crashed the whole process with a native access violation when parsing the v2 CatBoost models (3 of 8 fresh processes). A segfault would take down the Streamlit server for all users | Resolved | CatBoost models are explained with CatBoost's native TreeSHAP (8 of 8 clean); `shap.TreeExplainer` is used only for scikit-learn forests | `tests/test_risk_engine.py::test_explanations_do_not_crash_in_fresh_processes`, `::test_shap_values_add_up_to_the_explained_output` |
| R2 | Streamlit's markdown CSS shrank the KPI level word to 16px (measured in Edge), breaking the large-text contrast basis for `#CC6677` | Resolved | Card markup changed from `<p>` to `<div>` with three-class selectors | `scripts/check_layout.py` (28px/700 at 360, 390, 768, 1440 px) |
| R3 | At 390px the waterfall's labels and values clipped and the Plotly mode bar overlapped the chart | Resolved | Two-line labels, units in the axis title, padded x-range, mode bar hidden, explicit verified text colour | Edge screenshots at 390px |
| R4 | GEM's only trace *named* "Chaman Fault" is the Afghan segment (358 km from Chaman town) | Resolved (documented inference) | Pakistani segment identified as EMME `ME_PK209` by geometry and slip type; builder fails if the identification no longer holds | `tests/test_generate_data.py::test_chaman_segment_identification_holds`, `data/reference/SOURCES.md` |

### Professional UI pass (user request after the rebuild; no audit finding)

Each change was driven by a real-browser render (Edge via Playwright at 390px and 1440px) or by the dataviz palette validator, not by taste alone.

| ID | Change or finding | Evidence | Verified by |
|---|---|---|---|
| U1 | **Finding:** the brief's level palette fails the normal-vision colour-separation floor between Medium `#CC6677` and High `#CC3311` (ΔE 11.8 < 15). Level colours are kept only where the word and shape are printed (KPI tiles); the national map uses emphasis encoding (High vs slate `#64748B`, ΔE 22.5) | `validate_palette.js "#0077BB,#CC6677,#CC3311" --pairs all` → normal-vision FAIL | `tests/test_accessibility.py::test_map_uses_emphasis_encoding_not_the_level_palette` |
| U2 | Product header band, persistent banner, scenario chips, KPI stat tiles with a P(High) meter (High hue on every tile) and change chip | 27 colour pairs, all ≥ AA against composited surfaces | `tests/test_accessibility.py::test_contrast_meets_wcag_aa` |
| U3 | One hazard selector scopes both the SHAP explanation and a national map (dataviz rule: one filter above everything it scopes) | AppTest switches hazard; both panel titles follow | `tests/test_app_smoke.py::test_hazard_selector_scopes_explanation_and_map` |
| U4 | National map drawn without a basemap, over the real GEM fault and Natural Earth coastline layers: works offline and draws no contested boundaries; legend and table view always present | Edge render at 390px and 1440px | `tests/test_risk_engine.py::test_national_predictions_agree_with_single_district_predictions`, `::test_map_layers_are_real_reference_geometry` |
| U5 | District profile (scenario vs dataset value vs province median vs national range, Real/Synthetic source) moved to full width after the 7-column table overflowed the side panel | Edge render at 1440px | `tests/test_risk_engine.py::test_district_profile_marks_changes_and_sources` |
| U6 | Model card provenance shown as a pass/fail checklist covering every check, with mark + word + evidence | — | `tests/test_risk_engine.py::test_provenance_checklist_covers_every_check`, `tests/test_app_smoke.py::test_model_card_checklist_shows_every_check_passing` |
| U7 | Render fixes: header clipped by Streamlit's fixed toolbar (padding raised); exports moved after KPI tiles so phones show risk first; waterfall base/total values moved into row labels after colliding at 390px; asymmetric x-padding after a value label clipped; thin bars (width 0.55) and hairline grid per the dataviz mark spec | Before/after Edge screenshots | `scripts/check_layout.py` (no overflow, 28px/700 level words, 2 charts at 360/390/768/1440px) |

**Observed during the UI pass, not fixed:** raising summer temperature for Islamabad also lowers the seismic P(High) by 27.8 points. Temperature is not in the seismic label formula, so this is a non-physical relationship the model learned from 150 rows. Recommended fix: per-hazard feature sets or monotonic constraints (e.g. seismic score may only rise as fault distance falls), then retrain.

### Nigehban pass: input isolation, High recall, branding, publishing readiness

| ID | Change or finding | Evidence | Verified by |
|---|---|---|---|
| N1 | **Defect:** all three hazard models trained on one shared 16-column matrix (`train.py`: `V2_FEATURES` used for every target), so the seismic model responded to temperature (Islamabad seismic P(High) 0.943 → 0.665 when temperature moved 41.0 → 47.9 °C). Not a UI callback issue: the seismic model itself used temperature. **Fix:** per-hazard `HAZARD_FEATURES` with a justification per input; bundle schema v3 stores `feature_sets` and the engine passes each model only its own columns | Pre-fix measurement above; post-fix Edge check: Seismic card ■ +0.0 pts with temperature at maximum | `tests/test_risk_engine.py::test_hazard_models_never_receive_unrelated_inputs`, `::test_moving_unrelated_sliders_leaves_seismic_unchanged` |
| N2 | **Change:** recall-oriented High decision rule. Threshold on calibrated P(High) chosen on training folds (F2 for High), evaluated on pooled outer-fold predictions against argmax. Flood High recall 0.70 → 0.87 (false alarms 4 → 17); heatwave 0.65 → 0.91 (11 → 22); seismic 0.78 → 0.78 with 3 → 8 false alarms (no benefit, reported) | `models/metrics.json` → `per_class_held_out` | `tests/test_risk_engine.py::test_high_threshold_rule_flags_high_below_argmax`, `tests/test_train.py::test_threshold_choice_prefers_recall` |
| N3 | **Finding:** after isolation, temperature calibration made seismic probabilities worse (Brier 0.238 → 0.268, ECE 0.129 → 0.167); reported, not hidden | `models/metrics.json` | README calibration table |
| N4 | **Branding:** centered Nigehban header (flag-green band, inline SVG shield-and-eye emblem, Noto Nastaliq Urdu wordmark in `dir="rtl" lang="ur"` with line-height 2.6, Latin wordmark, "not an official NDMA or PDMA system" note). Crescent-and-star deliberately not used: it is a national state symbol. Font bundled under `static/fonts` (OFL) and served by Streamlit static serving | Edge: font loaded (`document.fonts.check` true), HTTP 200 for the font file; header, banner, district bar stacked in that order, sidebar in its own column | `tests/test_accessibility.py::test_nigehban_header_uses_rtl_nastaliq_and_no_state_symbols`, `tests/test_app_smoke.py::test_layout_order_brand_then_banner_then_district` |
| N5 | **Publishing:** local git repository and initial commit created; no GitHub remote or deployment exists yet (no GitHub CLI or credentials available to this session). README lists the exact remaining steps and does not claim a live link | `git log` | Not verifiable until pushed |

### Planning tools and favicon (user request; no audit finding)

| ID | Change | Evidence | Verified by |
|---|---|---|---|
| P1 | Climate stress test: uniform temperature shift and rainfall multiplier (defaults +2 °C, ×1.5; bounded 0–4 °C and ×0.5–2.0), re-scored with the deployed isolated models; newly-High list sorted by rise in P(High); current vs scenario maps; per-result extrapolation flag using the training range of the inputs each hazard actually uses; HYPOTHETICAL label | Defaults: 27 results become High (flood 6, heatwave 21, seismic 0); 5% of flood/heatwave results extrapolated | `tests/test_risk_engine.py::test_stress_test_leaves_seismic_untouched`, `::test_becomes_high_means_high_only_under_the_scenario`, `::test_stress_test_flags_extrapolation_beyond_training_range`, `::test_stress_settings_are_validated` |
| P2 | Multi-hazard hotspots: districts High for 2+ hazards, current or under the stress scenario, table plus map | 10 districts now, 15 under the default scenario | `tests/test_risk_engine.py::test_hotspots_are_districts_high_for_two_or_more_hazards`, `tests/test_app_smoke.py::test_planning_tools_render_with_hypothetical_labels` |
| P3 | Browser favicon: the header's SVG emblem on a flag-green tile, rendered to PNG by `scripts/build_favicon.py` and set as `page_icon` | Edge: tab title and icon link served by Streamlit | `tests/test_app_smoke.py::test_favicon_is_the_nigehban_emblem_png` |

### Layout pass and whole-project re-audit (2026-09-14)

| ID | Change or finding | Evidence | Verified by |
|---|---|---|---|
| L1 | **Change (user request):** page split into four numbered section bands (01 Risk overview, 02 Assessment detail, 03 Planning ahead, 04 Model and data), content in white panels, sidebar regrouped into brand, District, Climate, Land and exposure, and Scenario cards | Edge at 1440px and 390px: no horizontal overflow | `scripts/check_layout.py` (all viewports pass); 4 new colour pairs in `tests/test_accessibility.py::test_contrast_meets_wcag_aa` |
| A1 | **Defect (doc vs code):** F5.4 above says phones get "an on-page hint" for the collapsed sidebar; no hint existed in the code. **Fix:** phone-only note (CSS media query, ≤640px) at the top of section 01 | Edge: visible at 390px, hidden at 1440px | `tests/test_app_smoke.py::test_phone_hint_points_to_the_sidebar_controls` |
| A2 | **Defect (accessibility):** heading order started with a level-2 section title before the level-1 district title. **Fix:** Nigehban wordmark is the single h1, section titles h2, district bar h3 | Edge heading list after fix: NIGEHBAN, Risk overview, district, Assessment detail, … | `tests/test_app_smoke.py::test_heading_levels_never_skip` |
| A3 | **Defect (security):** a model pickle loaded with no trusted digest was reported only as an `info` provenance issue, so the page looked normal although `joblib.load` had executed an unverified file. **Fix:** raised to `critical`, which shows an error on the page | Unit test loads a copy without its manifest | `tests/test_risk_engine.py::test_unverified_model_is_a_critical_provenance_issue` |
| A4 | **Defect (docs):** README screenshots predated the Nigehban header and the layout pass. **Fix:** re-rendered from the running app in Edge | `docs/screenshots/*.png` | Inspected by eye |
| A5 | **Finding, not changed:** hyperparameter search (`train.py::tune`) maximises argmax macro-F1, but the deployed rule is the recall-tuned threshold, so model selection is not aligned with the deployed decision. Reported metrics stay honest (the rule is evaluated on outer folds); aligning the objective needs a retrain and changes every number above | `train.py` `tune()` objective | — |
| A6 | **Finding, not changed:** the deployed High threshold is chosen on out-of-fold predictions of hyperparameters that were tuned on all 150 rows, so the threshold itself is mildly optimistic. It does not affect any reported metric (nested CV picks its own thresholds per outer fold) | `train.py::train_all` | — |
| A7 | **Confirmed resolved:** the note after U7 (temperature moving seismic P(High)) was fixed by N1; left in place as the historical record | Edge: Gwadar at maximum temperature, Seismic card ■ +0.0 pts | `tests/test_risk_engine.py::test_moving_unrelated_sliders_leaves_seismic_unchanged` |

Re-audit checks that passed with no change: ruff lint and format on all 20 files; 123 tests including slow training tests; `generate_data.py --check` reproduces the committed CSV byte-for-byte; installed library versions equal the `requirements.txt` pins; README metric tables equal `models/metrics.json`; nested CV keeps the outer test fold out of tuning, calibration, and threshold choice; no secrets tracked by git; Edge session with district change and slider at maximum logged no console errors, failed requests, or exceptions.

### Requested items that cannot be completed from repository data

| Item | Why | Smallest real input needed |
|---|---|---|
| Historical baseline delta per district | No historical event dataset exists; `simulated_*` counts are synthetic | District-level event records, e.g. NDMA/PDMA situation reports or EM-DAT events geocoded to districts, with dates and hazard type |
| Official district list | GeoNames ADM2 is not the official list (e.g. Keamari, Hunza absent) | PBS 2023 census district list joined to OCHA COD-AB admin-2 boundaries (HDX `cod-ab-pak`) |
| Real climate and exposure features | Rainfall, temperature, NDVI, river distance, population density, infrastructure are synthetic | PMD station normals or WorldClim/CHIRPS zonal statistics; HydroRIVERS; PBS 2023 population and district areas |
| Seismic hazard beyond distance proxies | Distance-decay constants are judgment values | Probabilistic hazard values (e.g. PGA at 475-year return period) sampled at district locations |
