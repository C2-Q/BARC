"""Collect the final paper figures into outputs/figures/final_paper/ with clean names.

Run after the main pipeline and per-figure scripts have produced their outputs.
This step copies + renames each paper-facing figure to its manuscript name
and removes stale legacy artifacts so outputs/figures/final_paper/ contains
exactly the figures referenced by the paper.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import FIGURE_DIR, FINAL_PAPER_DIR

REVISED_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_paper_revised"

# (source stem without extension, destination stem under FINAL_PAPER_DIR).
# Each entry is published for both .pdf and .png if available.
FIGURE_PUBLISH_PLAN: list[tuple[Path, str]] = [
    (REVISED_DIR / "predictor_comparison_final_v2", "predictor_comparison"),
    (FINAL_PAPER_DIR / "incremental_predictive_gain_final", "incremental_predictive_gain"),
    (FINAL_PAPER_DIR / "structure_to_execution_chain_empirical_final", "structure_execution_chain"),
    (FINAL_PAPER_DIR / "lower_bound_vs_actual_final", "lower_bound_vs_actual"),
    (FIGURE_DIR / "qft_vs_other_real_traces", "qft_vs_real_traces"),
    (FINAL_PAPER_DIR / "real_trace_scaling_final", "real_trace_scaling"),
    (FINAL_PAPER_DIR / "qft_approximation_reduced_grid_final", "qft_approximation"),
    (FINAL_PAPER_DIR / "robustness_stochastic_routing_summary_final", "appendix_robustness_supply_routing"),
    (FINAL_PAPER_DIR / "lower_bound_gap_cases_final", "appendix_gap_cases"),
]

# Old stems under FINAL_PAPER_DIR to delete after publishing.
LEGACY_STEMS_TO_PURGE: list[str] = [
    "predictor_comparison_final",
    "predictor_comparison_final_v2",
    "incremental_predictive_gain_final",
    "structure_to_execution_chain_empirical_final",
    "lower_bound_vs_actual_final",
    "qft_vs_other_real_traces_final",
    "real_trace_scaling_final",
    "qft_approximation_reduced_grid_final",
    "robustness_stochastic_routing_summary_final",
    "lower_bound_gap_cases_final",
    "delta_max_vs_slowdown_final",
    "predictor_stability_slowdown_final",
    "predictor_stability_stall_final",
]


def publish_figures() -> list[Path]:
    FINAL_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    missing: list[Path] = []
    for source_stem, dest_stem in FIGURE_PUBLISH_PLAN:
        for extension in (".pdf", ".png"):
            source = source_stem.with_suffix(extension)
            if not source.exists():
                missing.append(source)
                continue
            destination = FINAL_PAPER_DIR / f"{dest_stem}{extension}"
            shutil.copy2(source, destination)
            published.append(destination)
            print(f"  publish: {destination.relative_to(PROJECT_ROOT)}")
    if missing:
        print("\nMissing source files (run the upstream generators first):", file=sys.stderr)
        for path in missing:
            print(f"  - {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    return published


def purge_legacy() -> None:
    for stem in LEGACY_STEMS_TO_PURGE:
        for extension in (".pdf", ".png"):
            path = FINAL_PAPER_DIR / f"{stem}{extension}"
            if path.exists():
                path.unlink()
                print(f"  remove legacy: {path.relative_to(PROJECT_ROOT)}")
    if REVISED_DIR.exists():
        shutil.rmtree(REVISED_DIR)
        print(f"  remove directory: {REVISED_DIR.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("Publishing paper figures into outputs/figures/final_paper/ ...")
    published = publish_figures()
    print(f"\nPublished {len(published)} files.")
    print("\nPurging legacy figure files...")
    purge_legacy()
    print("\nDone.")


if __name__ == "__main__":
    main()
