"""Audit F3.1: serving code must never use matplotlib/pyplot (process-global figure state).

Parses code with ``ast`` so docstrings and comments that *mention* pyplot are not
counted. ``import shap`` still imports pyplot as a library side effect; the
runtime half of this check (no figure ever created) is in test_app_smoke.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVING_MODULES = ("app.py", "risk_engine.py", "ui_theme.py")


@pytest.mark.parametrize("module", SERVING_MODULES)
def test_serving_code_does_not_use_matplotlib(module: str) -> None:
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offending += [a.name for a in node.names if a.name.split(".")[0] == "matplotlib"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "matplotlib":
            offending.append(node.module or "")
        elif isinstance(node, ast.Name) and node.id in {"plt", "pyplot", "matplotlib"}:
            offending.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr == "pyplot":
            offending.append("pyplot")
    assert offending == [], f"{module} uses matplotlib: {offending}"
