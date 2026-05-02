"""Rebuild the two real-workload figures (Fig. 6 and Fig. 7 in the paper draft)
to include the carry-lookahead adder, modular-arithmetic block, and QAOA MaxCut
families alongside the original ripple adder, multiplier, and QFT traces.

This script is self-contained: it does not modify the existing
`run_real_trace_scaling.py` or `run_qft_real_trace_scan.py` plot functions, so
the original real-trace pipeline remains intact. The figures are regenerated
at the existing repository paths (no `_final` suffix introduced):

  barc_stage1/outputs/figures/real_trace_scaling.{png,pdf}
  barc_stage1/outputs/figures/qft_vs_real_traces.{png,pdf}

The script reads:

  outputs/tables/real_trace_scaling_summary.csv          (adder + multiplier)
  outputs/tables/qft_real_trace_summary.csv              (QFT)
  outputs/tables/real_trace_workload_families_raw.csv    (CLA + modmult + QAOA)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plots import (
    PAPER_LABEL_SIZE,
    PAPER_TICK_SIZE,
    PAPER_TITLE_SIZE,
    _apply_paper_axis_style,
)
from src.utils import FIGURE_DIR, FINAL_PAPER_DIR, TABLE_DIR, ensure_output_dirs


# Colours chosen to match the existing real-trace style in the repo.
ADDER_COLOR = "#4c78a8"           # ripple adder
MULTIPLIER_COLOR = "#e45756"      # multiplier
QFT_COLOR = "#6a3d9a"             # QFT
CLA_COLOR = "#54a24b"             # CLA adder
MOD_MULT_COLOR = "#f58518"        # modular-arithmetic block
QAOA_COLOR = "#000000"            # QAOA

WORKLOAD_FAMILY_STYLE = {
    "ripple_adder": {"label": "Ripple", "color": ADDER_COLOR, "marker": "o"},
    "cla_adder": {"label": "CLA", "color": CLA_COLOR, "marker": "D"},
    "multiplier": {"label": "Multiplier", "color": MULTIPLIER_COLOR, "marker": "s"},
    "modular_multiplier": {"label": "Mod. arith. block", "color": MOD_MULT_COLOR, "marker": "P"},
    "qaoa_maxcut": {"label": "QAOA", "color": QAOA_COLOR, "marker": "^"},
    "exact_qft": {"label": "QFT", "color": QFT_COLOR, "marker": "X"},
}

# Bar-only palette used for the qft_vs_real_traces (Fig. 6) bar chart. Keyed by
# the display label so it stays decoupled from the line-plot colour scheme used
# in build_real_trace_scaling_figure (Fig. 7).
FIG6_BAR_COLORS = {
    "Ripple": "#8c8c8c",
    "CLA": "#2c7fb8",
    "Multiplier": "#6baed6",
    "Mod. arith. block": "#bdbdbd",
    "QAOA": "#e66101",
    "QFT": "#b35806",
}


def _load_workload_families_table() -> pd.DataFrame:
    path = TABLE_DIR / "real_trace_workload_families_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/run_real_trace_workload_families.py first"
        )
    return pd.read_csv(path)


def _representative_row(
    df: pd.DataFrame,
    workload_family: str,
    size_n: int,
    qaoa_p: int | None = None,
    graph_type: str | None = None,
    graph_seed: int | None = None,
) -> pd.Series | None:
    subset = df[(df["workload_family"] == workload_family) & (df["size_n"] == size_n)]
    if qaoa_p is not None:
        subset = subset[subset["qaoa_p"] == qaoa_p]
    if graph_type is not None:
        subset = subset[subset["graph_type"] == graph_type]
    if graph_seed is not None:
        subset = subset[subset["graph_seed"] == graph_seed]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_qft_vs_real_traces_figure(df: pd.DataFrame) -> Path:
    representative_n = 8
    rows: list[tuple[str, pd.Series | None]] = [
        ("ripple_adder", _representative_row(df, "ripple_adder", representative_n)),
        ("cla_adder", _representative_row(df, "cla_adder", representative_n)),
        ("multiplier", _representative_row(df, "multiplier", representative_n)),
        (
            "modular_multiplier",
            _representative_row(df, "modular_multiplier", representative_n),
        ),
        (
            "qaoa_maxcut",
            _representative_row(
                df,
                "qaoa_maxcut",
                representative_n,
                qaoa_p=2,
                graph_type="erdos_renyi_dense",
                graph_seed=0,
            ),
        ),
        ("exact_qft", _representative_row(df, "exact_qft", representative_n)),
    ]

    labels: list[str] = []
    slack_values: list[float] = []
    delta_values: list[float] = []
    colors: list[str] = []
    for family, row in rows:
        style = WORKLOAD_FAMILY_STYLE[family]
        if row is None or pd.isna(row.get("slack_ratio")):
            continue
        labels.append(style["label"])
        slack_values.append(float(row["slack_ratio"]))
        delta_values.append(float(row.get("mean_delta_max", math.nan)) if not pd.isna(row.get("mean_delta_max")) else 0.0)
        colors.append(FIG6_BAR_COLORS[style["label"]])

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))

    bar_kwargs = {"width": 0.62, "edgecolor": "#4d4d4d", "linewidth": 0.4}

    axes[0].bar(labels, slack_values, color=colors, **bar_kwargs)
    slack_max = max(slack_values, default=1.0)
    slack_label_offset = max(0.012, 0.02 * slack_max)
    for index, value in enumerate(slack_values):
        axes[0].text(
            index, value + slack_label_offset, f"{value:.2f}",
            ha="center", va="bottom", fontsize=7.2, color="#333333",
        )
    axes[0].set_ylim(0.0, min(1.05, slack_max + 0.16))
    axes[0].set_ylabel("Slack ratio", fontsize=PAPER_LABEL_SIZE)
    axes[0].set_title(
        f"Structural flexibility (n={representative_n})",
        fontsize=PAPER_TITLE_SIZE,
        pad=4,
    )
    _apply_paper_axis_style(axes[0], horizontal_grid=True)
    axes[0].tick_params(axis="x", labelsize=PAPER_TICK_SIZE)
    for tick_label in axes[0].get_xticklabels():
        tick_label.set_rotation(25)
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")

    axes[1].bar(labels, delta_values, color=colors, **bar_kwargs)
    if delta_values:
        max_delta = max(delta_values)
        axes[1].set_yscale("symlog", linthresh=1.0)
        axes[1].set_ylim(0.0, max(max_delta * 4.0, 10.0))
        for index, value in enumerate(delta_values):
            if value <= 0:
                axes[1].text(
                    index, 1.2, "0",
                    ha="center", va="bottom", fontsize=7.2, color="#333333",
                )
            else:
                label_y = value * 1.6
                text = f"{value:.1f}" if value < 100 else f"{value:,.0f}"
                axes[1].text(
                    index, label_y, text,
                    ha="center", va="bottom", fontsize=7.2, color="#333333",
                )
    axes[1].set_ylabel(r"Mean $\Delta_{\max}$ (symlog)", fontsize=PAPER_LABEL_SIZE)
    axes[1].set_title(
        f"Delivery pressure (n={representative_n})",
        fontsize=PAPER_TITLE_SIZE,
        pad=4,
    )
    _apply_paper_axis_style(axes[1], horizontal_grid=True)
    axes[1].tick_params(axis="x", labelsize=PAPER_TICK_SIZE)
    for tick_label in axes[1].get_xticklabels():
        tick_label.set_rotation(25)
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")

    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.30, top=0.88, wspace=0.28)

    path = FIGURE_DIR / "qft_vs_real_traces.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    FINAL_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    final_path = FINAL_PAPER_DIR / "qft_vs_real_traces.png"
    figure.savefig(final_path, dpi=300, bbox_inches="tight")
    figure.savefig(final_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def build_real_trace_scaling_figure(df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

    # Arithmetic + modular only — QAOA is intentionally omitted from Fig. 7
    # because (i) its mean Delta_max is 2-4 orders of magnitude larger than the
    # arithmetic families and would compress the arithmetic curves into a
    # near-horizontal band, and (ii) it uses a different (reduced) bounded-
    # delivery grid and a different size set. QAOA appears in Fig. 6 and the
    # representative summary table.
    families = ["ripple_adder", "cla_adder", "multiplier", "modular_multiplier"]
    for family in families:
        style = WORKLOAD_FAMILY_STYLE[family]
        subset = df[df["workload_family"] == family].sort_values("size_n")
        if subset.empty:
            continue
        xs = subset["size_n"].to_numpy()
        slack = subset["slack_ratio"].to_numpy()
        delta = subset["mean_delta_max"].to_numpy()

        axes[0].plot(
            xs, slack,
            marker=style["marker"], color=style["color"],
            linewidth=1.7, markersize=7, label=style["label"],
        )
        axes[1].plot(
            xs, delta,
            marker=style["marker"], color=style["color"],
            linewidth=1.7, markersize=7, label=style["label"],
        )

    axes[0].set_xlabel("Problem size $n$")
    axes[0].set_ylabel("Slack ratio")
    axes[0].set_title("Structural flexibility")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, loc="center right")

    axes[1].set_xlabel("Problem size $n$")
    axes[1].set_ylabel(r"Mean $\Delta_{\max}$ (symlog)")
    axes[1].set_title("Delivery pressure")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].legend(fontsize=9, loc="upper left")

    path = FIGURE_DIR / "real_trace_scaling.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    FINAL_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    final_path = FINAL_PAPER_DIR / "real_trace_scaling.png"
    figure.savefig(final_path, dpi=300, bbox_inches="tight")
    figure.savefig(final_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    ensure_output_dirs()
    df = _load_workload_families_table()
    print(f"loaded {len(df)} rows from real_trace_workload_families_raw.csv")
    qft_path = build_qft_vs_real_traces_figure(df)
    scaling_path = build_real_trace_scaling_figure(df)
    print(f"saved: {qft_path} (and .pdf)")
    print(f"saved: {scaling_path} (and .pdf)")


if __name__ == "__main__":
    main()
