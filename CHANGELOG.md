# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/). Every change is traced to a finding in
[AUDIT.md](AUDIT.md).

## [2.0.0] - 2026-09-14

Full audited rebuild of v1 (v1 files are kept in `archive/v1/`).

### Added

- **Data:** real district coordinates, elevation, active-fault and Makran subduction distances, and coastline
  (GeoNames, GEM, Natural Earth).
- **Validation:** nested cross-validation with class weighting and temperature calibration, reported with Brier
  score, log loss, ROC-AUC and quadratic kappa.
- **Models:** per-hazard input isolation, so the seismic score cannot react to temperature or rainfall.
- **Decision rule:** a recall-oriented High rule, with thresholds chosen on training folds (F2) and before/after
  per-class metrics.
- **Explanations:**
  - SHAP waterfall and a one-sentence insight per hazard.
  - National map drawn over real fault lines.
  - District profile with national range, rank and province comparison.
- **Planning tools:** climate stress test (temperature shift and rainfall multiplier, with extrapolation flags) and
  multi-hazard hotspots.
- **Trust checks:** start-up provenance checks (model SHA-256, training data hash, library versions, calibration)
  shown as a checklist. A missing digest is a critical, visible error.
- **Dashboard:** "Nigehban" design with Nastaliq Urdu wordmark, speedometer KPI gauges, per-section accent colours,
  CSV and text report export, and a phone layout.
- **Accessibility:** 93 WCAG AA colour pairs verified in tests; shapes accompany every level colour.
- **Engineering:** CI (lint, data reproducibility, tests, training smoke run), pinned Python 3.12 locks, and
  community health files.

### Fixed

- **Model and data mismatch:** the shipped model had been trained on a different dataset than the one shown.
- **Inflated scores:** v1 reported scores from selecting and tuning on the same folds.
- **Misleading sliders:** what-if sliders could leave the training range without any warning.
