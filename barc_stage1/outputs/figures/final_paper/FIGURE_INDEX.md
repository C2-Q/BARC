# Final Paper Figure Index

This folder contains only the figures selected for the current paper draft and appendix.

## Main Text

- `predictor_comparison_final`: predictor ranking across the three evaluation tasks
- `incremental_predictive_gain_final`: incremental gain from adding slack ratio and `Delta_max`
- `structure_to_execution_chain_empirical_final`: empirical structure -> system -> execution chain
- `lower_bound_vs_actual_final`: lower-bound validation against executed makespan
- `qft_vs_other_real_traces_final`: real workloads across structural regimes
- `real_trace_scaling_final`: adder/multiplier scaling under bounded delivery
- `qft_approximation_reduced_grid_final`: exact vs. approximate QFT under bounded delivery

## Appendix

- `predictor_stability_stall_final`: subgroup stability for stall prediction (AUC)
- `predictor_stability_slowdown_final`: subgroup stability for slowdown prediction (Spearman `|rho|`)
- `lower_bound_gap_cases_final`: representative positive-gap mechanisms

## Consistency Notes

- The old combined file `predictor_stability_heatmap_final` has been retired in favor of the two clearer appendix figures above.
- The old exploratory file `delta_max_vs_slowdown_final` is not part of the current paper figure set and is intentionally excluded.
- If the LaTeX source still references `predictor_stability_heatmap_final`, update it to use `predictor_stability_stall_final` and `predictor_stability_slowdown_final`.
