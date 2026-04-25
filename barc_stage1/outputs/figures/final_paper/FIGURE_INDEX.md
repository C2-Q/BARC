# Final Paper Figure Index

This folder holds the nine figures referenced by the manuscript. File names
match the LaTeX `\includegraphics{figures/final_paper/<name>.pdf}` paths.

## Main-Text Figures

| File | Manuscript label |
| --- | --- |
| `predictor_comparison.pdf` | `fig:predictor_comparison` |
| `incremental_predictive_gain.pdf` | `fig:incremental_gain` |
| `structure_execution_chain.pdf` | `fig:empirical_chain` |
| `lower_bound_vs_actual.pdf` | `fig:lower_bound` |
| `qft_vs_real_traces.pdf` | `fig:qft_vs_real` |
| `real_trace_scaling.pdf` | `fig:real_scaling` |
| `qft_approximation.pdf` | `fig:qft_approx` |

## Appendix Figures

| File | Manuscript label |
| --- | --- |
| `appendix_robustness_supply_routing.pdf` | `fig:appendix_robustness_supply_routing` |
| `appendix_gap_cases.pdf` | `fig:appendix_gap_cases` |

## Reproducing

The single command that regenerates and republishes every figure in this
folder is documented in the repository root `README.md` (see the
*Reproducibility* section). Internally it runs the upstream generator
scripts and then `scripts/publish_paper_figures.py`, which copies each
figure here under its manuscript name.

## Other Figures

`figures/delta_max_illustration.{pdf,png}` (one level up) is referenced
separately as the project's key illustrative figure on the README and
remains outside this folder.
