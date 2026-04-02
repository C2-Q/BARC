from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "barc_stage1"
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"

FIGURE_FILES = [
    ("figures/delta_max_illustration.pdf", OUTPUT_ROOT / "figures" / "delta_max_illustration.pdf"),
    ("figures/final_paper/predictor_comparison_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "predictor_comparison_final.pdf"),
    ("figures/final_paper/incremental_predictive_gain_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "incremental_predictive_gain_final.pdf"),
    (
        "figures/final_paper/structure_to_execution_chain_empirical_final.pdf",
        OUTPUT_ROOT / "figures" / "final_paper" / "structure_to_execution_chain_empirical_final.pdf",
    ),
    ("figures/final_paper/delta_max_vs_slowdown_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "delta_max_vs_slowdown_final.pdf"),
    ("figures/final_paper/lower_bound_vs_actual_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "lower_bound_vs_actual_final.pdf"),
    ("figures/final_paper/qft_vs_other_real_traces_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "qft_vs_other_real_traces_final.pdf"),
    ("figures/final_paper/real_trace_scaling_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "real_trace_scaling_final.pdf"),
    (
        "figures/final_paper/qft_approximation_reduced_grid_final.pdf",
        OUTPUT_ROOT / "figures" / "final_paper" / "qft_approximation_reduced_grid_final.pdf",
    ),
    (
        "figures/final_paper/robustness_stochastic_routing_summary_final.pdf",
        OUTPUT_ROOT / "figures" / "final_paper" / "robustness_stochastic_routing_summary_final.pdf",
    ),
    ("figures/final_paper/lower_bound_gap_cases_final.pdf", OUTPUT_ROOT / "figures" / "final_paper" / "lower_bound_gap_cases_final.pdf"),
]

TABLE_FILES = [
    ("tables/family_summary.csv", OUTPUT_ROOT / "tables" / "family_summary.csv"),
    ("tables/inversion_summary.csv", OUTPUT_ROOT / "tables" / "inversion_summary.csv"),
    ("tables/predictive_classification_summary.csv", OUTPUT_ROOT / "tables" / "predictive_classification_summary.csv"),
    ("tables/predictive_regression_summary.csv", OUTPUT_ROOT / "tables" / "predictive_regression_summary.csv"),
    ("tables/predictive_multivariate_regression.csv", OUTPUT_ROOT / "tables" / "predictive_multivariate_regression.csv"),
    ("tables/incremental_predictive_models.csv", OUTPUT_ROOT / "tables" / "incremental_predictive_models.csv"),
    ("tables/bootstrap_slack_vs_tdepth.csv", OUTPUT_ROOT / "tables" / "bootstrap_slack_vs_tdepth.csv"),
    ("tables/causal_chain_correlations.csv", OUTPUT_ROOT / "tables" / "causal_chain_correlations.csv"),
    ("tables/lower_bound_validation.csv", OUTPUT_ROOT / "tables" / "lower_bound_validation.csv"),
    ("tables/lower_bound_gap_cases.csv", OUTPUT_ROOT / "tables" / "lower_bound_gap_cases.csv"),
    ("tables/qft_real_trace_summary.csv", OUTPUT_ROOT / "tables" / "qft_real_trace_summary.csv"),
    ("tables/real_trace_scaling_summary.csv", OUTPUT_ROOT / "tables" / "real_trace_scaling_summary.csv"),
    ("tables/qft_approximation_reduced_grid_summary.csv", OUTPUT_ROOT / "tables" / "qft_approximation_reduced_grid_summary.csv"),
    ("tables/stochastic_supply_ranking_summary.csv", OUTPUT_ROOT / "tables" / "stochastic_supply_ranking_summary.csv"),
    ("tables/routing_proxy_ranking_summary.csv", OUTPUT_ROOT / "tables" / "routing_proxy_ranking_summary.csv"),
    ("tables/predictive_analysis_summary.txt", OUTPUT_ROOT / "tables" / "predictive_analysis_summary.txt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean manuscript asset bundle for submission.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "submission_bundle",
        help="Directory where the manuscript-facing bundle will be written.",
    )
    return parser.parse_args()


def _copy_bundle_items(items: list[tuple[str, Path]], destination_root: Path, section: str) -> list[str]:
    manifest_lines = [f"## {section}", ""]
    for relative_target, source_path in items:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required submission asset: {source_path}")
        target_path = destination_root / relative_target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        manifest_lines.append(f"- `{relative_target}` <- `{source_path.relative_to(REPO_ROOT)}`")
    manifest_lines.append("")
    return manifest_lines


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# Submission Bundle",
        "",
        "This directory contains the manuscript-facing PDF figures and supporting tables",
        "for the QCE submission `When Depth Misleads: Predictive Modeling of Quantum",
        "Execution Stalls under Bounded Magic-State Delivery`.",
        "",
        "The `figures/` subtree is laid out to match the LaTeX `\\includegraphics{figures/...}` paths.",
        "",
    ]
    manifest_lines.extend(_copy_bundle_items(FIGURE_FILES, output_dir, "Figures"))
    manifest_lines.extend(_copy_bundle_items(TABLE_FILES, output_dir, "Tables"))

    manifest_path = output_dir / "MANIFEST.md"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print(f"submission bundle written to: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
