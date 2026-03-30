# LaTeX Sync Notes

This folder is now aligned to the intended paper figure set.

## Required LaTeX fixes

### 1. Replace the retired appendix stability figure

The old file `predictor_stability_heatmap_final` has been removed.
Use the two clearer appendix figures below instead:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/final_paper/predictor_stability_stall_final.png}
  \caption{Predictor stability across subgroup analyses for stall prediction (AUC). Across family-level and supply-regime slices, $\Delta_{\max}$ remains the leading predictor in nearly all informative subsets, while slack ratio typically matches or exceeds T-depth among the structure-aware metrics.}
  \label{fig:appendix_stability_stall}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/final_paper/predictor_stability_slowdown_final.png}
  \caption{Predictor stability across subgroup analyses for slowdown prediction (Spearman $|\rho|$). Across family-level and supply-regime slices, $\Delta_{\max}$ remains the leading predictor in nearly all informative subsets, while slack ratio typically matches or exceeds T-depth among the structure-aware metrics.}
  \label{fig:appendix_stability_slowdown}
\end{figure*}
```

### 2. Remove the duplicated incremental-gain figure block

The current paper text includes `incremental_predictive_gain_final` twice with the same label `fig:incremental_gain`.
Keep only one copy.

### 3. Ensure the concept illustration exists

The main-text reference

```tex
\includegraphics[width=0.95\columnwidth]{figures/delta_max_illustration.pdf}
```

is now backed by:

- `../delta_max_illustration.pdf`
- `../delta_max_illustration.png`

## Selected figure set

### Main text

- `predictor_comparison_final`
- `incremental_predictive_gain_final`
- `structure_to_execution_chain_empirical_final`
- `lower_bound_vs_actual_final`
- `qft_vs_other_real_traces_final`
- `real_trace_scaling_final`
- `qft_approximation_reduced_grid_final`

### Appendix

- `predictor_stability_stall_final`
- `predictor_stability_slowdown_final`
- `lower_bound_gap_cases_final`
