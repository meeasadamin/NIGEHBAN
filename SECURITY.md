# Security policy

## Supported versions

Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please **do not** open a public issue. Report privately through GitHub:
[Security → Report a vulnerability](https://github.com/meeasadamin/NIGEHBAN/security/advisories/new).

Include what is affected, how to reproduce it, and the impact you expect. You should get a first reply within
7 days.

## Scope notes

- **Model loading.** `models/best_hazard_pipeline.pkl` is a pickle, and loading a pickle runs code. `risk_engine.load_bundle`
  checks its SHA-256 before unpickling and refuses a mismatch. If no trusted digest is available, the dashboard shows a
  critical error on the page. For deployments, set the `NDMA_MODEL_SHA256` secret so the digest does not live next to
  the file it protects. Never load a model bundle from an untrusted source.
- **Dashboard.** The app accepts no uploads, escapes all dynamic text before rendering HTML, and hides tracebacks from
  users (see `.streamlit/config.toml`).
- **Out of scope.** Prediction quality on real-world events. All risk labels are synthetic and the app is not an
  operational system.
