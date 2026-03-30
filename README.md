# BARC Stage 1 Artifact

## Overview

This repository accompanies a paper on bounded T-state delivery in fault-tolerant quantum execution. The artifact studies how circuit dependency structure interacts with a fixed delivery rate `C` and a fixed buffer capacity `B`.

The main outputs of the artifact are:

- a structural metric based on T-node slack
- a system-level metric `delta_max`
- a fixed-schedule lower bound on executable makespan

The repository is organized around four layers:

- model: dependency DAGs and demand traces
- scheduling: static schedule construction
- simulation: bounded-delivery execution
- analysis: predictive evaluation and figure generation

## Key Concepts

- `slack_ratio`: fraction of T nodes with positive slack in the dependency DAG
- `mean_t_slack`: mean slack over T nodes
- `delta_max`: peak prefix deficit between cumulative T demand and cumulative delivery capacity
- `slowdown_ratio`: `T_exe / T_static`

For a fixed valid schedule with static depth `T_static`, delivery capacity `C`, buffer `B`, and prefix deficit `delta_max`, the artifact evaluates the lower bound

`T_static + max(0, delta_max - B) / C`

against the simulated executed makespan `T_exe`.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full paper pipeline:

```bash
python scripts/run_qce_paper.py
```

Run the slack tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The main pipeline generates:

- paper-facing tables in `outputs/tables/`
- paper-facing figures in `outputs/figures/`
- secondary tables and figures in `outputs/appendix/`

## Main Results

The primary figures for the paper are:

- [slack_vs_slowdown.png](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/figures/slack_vs_slowdown.png)
- [delta_max_vs_slowdown.png](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/figures/delta_max_vs_slowdown.png)
- [slack_vs_stall.png](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/figures/slack_vs_stall.png)
- [structure_to_execution_chain.png](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/figures/structure_to_execution_chain.png)
- [lower_bound_vs_actual.png](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/figures/lower_bound_vs_actual.png)

The primary tables are:

- [stage1_grid_scan.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/stage1_grid_scan.csv)
- [family_summary.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/family_summary.csv)
- [predictive_classification_summary.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/predictive_classification_summary.csv)
- [predictive_regression_summary.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/predictive_regression_summary.csv)
- [predictive_multivariate_regression.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/predictive_multivariate_regression.csv)
- [causal_chain_summary.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/causal_chain_summary.csv)
- [causal_chain_correlations.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/causal_chain_correlations.csv)
- [lower_bound_validation.csv](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/lower_bound_validation.csv)
- [predictive_analysis_summary.txt](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/predictive_analysis_summary.txt)
- [paper1_reframing_summary.txt](/Users/mac/Documents/GitHub/BARC/barc_stage1/outputs/tables/paper1_reframing_summary.txt)

The real-trace grounding remains available under `src/real_trace/` and `outputs/real_trace/`. In the paper narrative, these traces serve as grounding examples rather than as the main source of method comparison.

## Scope and Limitations

The artifact uses:

- deterministic delivery capacity
- deterministic buffer capacity
- dependency-aware synthetic DAG families
- fixed schedule policies

The artifact does not include:

- routing or layout effects
- stochastic delivery
- noise, decoding, or full physical simulation
- optimal scheduling over all valid schedules

The lower bound is a fixed-schedule result. It does not characterize optimal scheduling across all valid schedules and it does not minimize `delta_max` over the full schedule space.

## Repository Layout

The source tree keeps the paper pipeline compact:

- `src/dag.py`, `src/trace.py`: model layer
- `src/schedule.py`: scheduling layer
- `src/simulator.py`: simulation layer
- `src/metrics.py`, `src/predictive_analysis.py`: analysis layer
- `src/plots.py`: figure generation
- `src/real_trace/`: real-trace grounding utilities
- `scripts/run_qce_paper.py`: single entry point
- `tests/`: slack validation tests

## Appendix Outputs

Non-primary outputs are written to `outputs/appendix/`. This includes:

- policy comparison artifacts
- QTV tradeoff artifacts
- family-level gain boxplots

These files are retained for completeness, but they are not part of the main paper narrative.
