# LaTeX Sync Notes

The paper draft uses `\includegraphics{figures/final_paper/<name>.pdf}`
paths; the artifact stages those exact filenames in this folder.

## Recommended Workflow

1. From the artifact root, run the reproduce sequence documented in the
   repository `README.md` *Reproduce the paper figures* section. The final
   step, `python scripts/publish_paper_figures.py`, copies and renames each
   figure here under its manuscript name.
2. Optional: assemble a clean manuscript bundle for handoff:
   `python barc_stage1/scripts/prepare_submission_bundle.py`. The resulting
   `submission_bundle/figures/` subtree mirrors the LaTeX include paths.
3. Copy or symlink `submission_bundle/figures/` (or this folder directly)
   into the LaTeX project so the paper sees the expected paths.

## Figure -> Manuscript Label

See `FIGURE_INDEX.md` in this folder for the full table.

## Scope

This note only covers manuscript-facing figure paths. Numerical claim
tracing is documented in the repository root `README.md` and in the
generated `submission_bundle/MANIFEST.md`.
