# v1 archive

Unmodified v1 artifacts, kept so every finding in `../../AUDIT.md` can be reproduced. Nothing here is used by the v2 pipeline or app. The v2 app refuses to load `best_hazard_pipeline_v1.pkl` (schema version check).

| File | What it is | SHA-256 |
|---|---|---|
| `app_v1.py` | v1 Streamlit app (audit line references point here) | `9c8c1df78ba4e89580c148b5d228637a323b5db01ef9e291f008a605d47d0cc4` |
| `generate_data_v1.py` | v1 generator | `c8f934a5dea129e88032be8f7f9c706f96057dde8f6fae30203c05edb1226c4f` |
| `pakistan_districts_v1.csv` | v1 dataset (reproducible from `generate_data_v1.py`) | `3c00920edb1e7fccbc1c60ae1789abb92f1df584521866721284d0209299f694` |
| `best_hazard_pipeline_v1.pkl` | v1 shipped model bundle, fitted on different data than the CSV (audit F1.1) | `03231ac3bee2d0ba5dc2db2c5c584ab06018b36fceaf57c3cfe25c3e3adb6f6f` |
| `requirements_v1.txt` | v1 requirements (audit F6.1) | `1b47805d8813db8a570070d6c5b79bbf4b1555c1aa26227cae7664809cbec921` |
| `README_v1.md` | v1 README, including the metrics table that disagreed with the bundle (audit F1.2) | `8e8c11e6def37ed2f5e877e14b930f0752728ffe8cfa589d430647198f49e4a9` |
| `images/` | EDA figures written by `analysis.ipynb` on v1 data (red-green `RdYlGn_r` colormap, audit F5.3) | not individually listed |
| `v1_nested_cv_metrics.json` | Output of `scripts/evaluate_v1_protocol.py`: v2's nested-CV protocol applied to v1 data | generated |

The pickle executes code when loaded. Only load it in a disposable environment, after checking its hash against this table.
