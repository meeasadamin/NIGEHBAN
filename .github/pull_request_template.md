## What and why

<!-- One or two sentences: what changes, and the problem it solves. Link the issue: Closes #123 -->

## How it was verified

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `python generate_data.py --check` passes
- [ ] `pytest -m "not slow"` passes (and `pytest -m slow` if training code changed)
- [ ] UI changes: checked at phone (390px) and desktop (1440px) widths; new colour pairs added to `ui_theme.contrast_requirements()`
- [ ] Model or data changes: retrained with `python train.py`, and README/AUDIT numbers updated from `models/metrics.json`

## Notes for reviewers

<!-- Trade-offs, follow-ups, or anything that behaves differently from before. -->
