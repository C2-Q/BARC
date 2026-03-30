from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt


STATIC_COLOR = "#1f77b4"
CAPACITY_AWARE_COLOR = "#ff7f0e"
SMOOTHED_COLOR = "#d62728"
GRID_ALPHA = 0.18
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 12
ANNOTATION_SIZE = 11


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "savefig.facecolor": "white",
        }
    )


def apply_axis_style(axis) -> None:
    axis.grid(True, alpha=GRID_ALPHA)
    axis.tick_params(axis="both", labelsize=TICK_SIZE)


def save_figure_bundle(figure, output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix.lower() not in {".png", ".pdf"}:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        figure.tight_layout()
        figure.savefig(png_path, dpi=300)
        figure.savefig(pdf_path)
    plt.close(figure)
    return png_path
