from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FINAL_PAPER_DIR = FIGURE_DIR / "final_paper"
TABLE_DIR = OUTPUT_ROOT / "tables"
APPENDIX_DIR = OUTPUT_ROOT / "appendix"


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for path in (OUTPUT_ROOT, FIGURE_DIR, FINAL_PAPER_DIR, TABLE_DIR, APPENDIX_DIR):
        path.mkdir(parents=True, exist_ok=True)


def mean_int(values: Iterable[int]) -> float:
    """Return the arithmetic mean of integer-like values as a float."""
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def normalize_metric_name(name: str) -> str:
    """Map legacy export names onto the paper-facing metric vocabulary."""
    normalized = name
    replacements = (
        ("compressibility_mean_t_slack", "mean_t_slack"),
        ("compressibility_slack_ratio", "slack_ratio"),
        ("compressibility_legacy", "legacy_compressibility"),
        ("compressibility", "legacy_compressibility"),
        ("Delta_max", "delta_max"),
        ("Gamma", "gamma"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def normalize_output_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with paper-facing metric names in columns and metric labels."""
    normalized = dataframe.copy()
    normalized.columns = [normalize_metric_name(column) for column in normalized.columns]
    if "metric" in normalized.columns:
        normalized["metric"] = normalized["metric"].astype(str).map(normalize_metric_name)
    if "left_metric" in normalized.columns:
        normalized["left_metric"] = normalized["left_metric"].astype(str).map(normalize_metric_name)
    if "right_metric" in normalized.columns:
        normalized["right_metric"] = normalized["right_metric"].astype(str).map(normalize_metric_name)
    if "feature" in normalized.columns:
        normalized["feature"] = normalized["feature"].astype(str).map(normalize_metric_name)
    return normalized
