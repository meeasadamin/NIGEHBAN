# Contributing to Nigehban

Thanks for your interest. This is a portfolio demonstrator, so the bar for changes is that they keep the project
**honest** (numbers traceable to `models/metrics.json`), **reproducible** (seeded data and pinned dependencies), and
**accessible** (verified contrast, no colour-only meaning).

## Set up

Python 3.12 is required.

```bash
git clone https://github.com/meeasadamin/NIGEHBAN.git
cd NIGEHBAN
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
python generate_data.py --check      # the committed CSV must still reproduce from its seed
pytest -m "not slow"                 # about 1 minute
pytest -m slow                       # training smoke run, if you touched train.py or risk_engine.py
```

CI runs the same steps on every pull request.

## Guidelines

- **Code layout.** UI layout goes in `app.py`; colours, CSS and HTML fragments in `ui_theme.py`; data, models, SHAP
  and exports in `risk_engine.py`. Keep `app.py` free of model logic.
- **Colours.** Every new text or graphic colour pair must be added to `ui_theme.contrast_requirements()`. The tests
  enforce WCAG AA against the surface actually drawn.
- **Dynamic HTML.** Build it through the `ui_theme` helpers, which escape plain strings. Never pass user or data
  text to `unsafe_allow_html` unescaped.
- **Model or data changes.** Retrain with `python train.py`, commit the new bundle, manifest and metrics together,
  and update the numbers in `README.md` from `models/metrics.json`. Record why in `AUDIT.md`.
- **Dependencies.** Edit `requirements.in` / `requirements-dev.in`, then recompile the locks:
  `uv pip compile requirements.in --python-version 3.12 --universal -o requirements.txt` (same for dev).
- **Commits.** Use short imperative messages, e.g. `fix: clip range bar position`.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
