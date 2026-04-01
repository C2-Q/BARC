# LaTeX Sync Notes

The paper draft uses `\includegraphics{figures/...}` paths, while the artifact generates figures under `barc_stage1/outputs/figures/`.

## Recommended Workflow

1. Rebuild artifact outputs from the repository root:
   `cd barc_stage1 && python scripts/run_qce_paper.py`
2. Prepare a manuscript-facing asset bundle:
   `python barc_stage1/scripts/prepare_submission_bundle.py`
3. Copy or symlink the generated `submission_bundle/figures/` subtree into the LaTeX project so that the paper sees the expected `figures/...` paths.

## Current Figure Mapping

- `figures/delta_max_illustration.pdf` comes from `barc_stage1/outputs/figures/delta_max_illustration.pdf`
- `figures/final_paper/*.pdf` comes from `barc_stage1/outputs/figures/final_paper/*.pdf`

## Scope

This note only covers the manuscript-facing figure paths. Numerical claim tracing is documented in the repository root `README.md` and in the generated `submission_bundle/MANIFEST.md`.
