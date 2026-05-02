# Paper Text Patches — Workload-Family Extension

This document collects the LaTeX snippets the workload-family branch implies for
the paper. The repository does not contain the LaTeX source; paste these into
the externally maintained manuscript. The σ_quota promotion is **not** included
here — that lives on the `promote-sigma-quota` follow-up branch.

## Wording rules used throughout these patches

The following terminology is used consistently in the patches below; please
preserve it when pasting into the manuscript.

- **compressibility families** — the high/medium/low constructed DAG families.
- **compressibility-family evaluation** — the 4,904-finite-instance controlled
  evaluation. Scope the 4,904 claim explicitly to this evaluation.
- **workload families** — the full evaluation set, including the
  compressibility families and the circuit workloads.
- **circuit workloads** — ripple adder, CLA adder, integer multiplier,
  modular-arithmetic block, QAOA MaxCut, exact QFT.
- The new families are **first-class workload families in the main workload
  spectrum**, not "external validation".
- Avoid the word "synthetic" in paper-facing text (use "compressibility-family"
  or "constructed-DAG instance" if you need to disambiguate).
- The modular-arithmetic family is named **"Modular-arithmetic block;
  structural surrogate for modular multiplication patterns"** in any role
  description. Do **not** call it a "modular multiplier" — the implementation
  on this branch is a controlled add-subtract-modulus chain and does not
  include the conditional add-back step that a verified mod-N multiplier
  requires. The compact label in tables/figures is **Mod. arith. block**.

## Repo-to-LaTeX figure path mapping

The repository generates figures under `barc_stage1/outputs/figures/`. The
LaTeX manuscript references files under `figures/final_paper/` with `_final`
suffixes. The repo paths are intentionally **not** changed to match the LaTeX
paths; the packaging step is responsible for the copy/rename. The mapping is:

```
barc_stage1/outputs/figures/qft_vs_real_traces.pdf
    -> figures/final_paper/qft_vs_other_real_traces_final.pdf

barc_stage1/outputs/figures/real_trace_scaling.pdf
    -> figures/final_paper/real_trace_scaling_final.pdf
```

The packaging script (`prepare_submission_bundle.py` /
`publish_paper_figures.py`) is responsible for the copy with rename. No new
`_final`-suffixed files are introduced by this branch.

## Aggregation and policy used in the new tables/figures

All headline numbers in the new representative table, Fig. 6, and Fig. 7 use
the **σ_static** scheduling policy (`policy == "static_min"` in the per-instance
result CSVs). Per-instance result CSVs at
`barc_stage1/outputs/real_trace/<instance_id>_results.csv` contain the full
per-(C, B, policy) grid for σ_static, σ_ca, and σ_smooth; the
`real_trace_workload_families_*` summary CSVs project onto σ_static and
average Δ_max over unique C values (since static-schedule Δ_max depends only
on C, not B).

This matches the existing real-trace pipeline:
`scripts/run_real_trace_scaling.py` and `scripts/run_qft_real_trace_scan.py`
also filter to σ_static and average over unique C values.

The captions below state this explicitly so the reader knows which policy is
being reported.

---

## Abstract

**Replace:**

```
with arithmetic circuits and exact quantum Fourier transform (QFT) traces providing additional empirical grounding
```

**With:**

```
with arithmetic, modular-arithmetic, QAOA MaxCut, and quantum Fourier transform (QFT) circuit workloads drawn from a workload spectrum that complements the compressibility-family evaluation
```

**Replace:**

```
Across 4{,}904 finite instances, the lower bound shows zero observed violations, with 88.9\% of instances falling within one cycle of the bound.
```

**With:**

```
Across 4{,}904 finite instances of the compressibility-family evaluation, the lower bound shows zero observed violations and is tight in most aggregate cases, although tightness degrades when backlog persists across multiple logical steps.
```

(Numerical values inside the second sentence are unchanged on this branch — the
σ_quota branch will recompute them.)

---

## Contribution list — second item

**Replace with:**

```
\item We show that $\Delta_{\max}$ is the strongest schedule-level indicator of \textbf{execution stalls and makespan slowdown} in our evaluated instances, and validate this relationship across a workload spectrum that spans serial arithmetic, parallel arithmetic, modular-arithmetic, QAOA MaxCut, and QFT circuit workloads.
```

---

## Table I — workload list

Add three rows to the workload list. The role descriptions are deliberately
conservative and do not overclaim:

```latex
Circuit workload & CLA adder            & $n \in \{4,8,12,16\}$ & Parallel arithmetic family \\
Circuit workload & Mod. arith. block    & $n \in \{4,6,8\}$ & Modular-arithmetic block; structural surrogate for modular multiplication patterns \\
Circuit workload & QAOA MaxCut          & $n \in \{6,8,10\}$; $p \in \{1,2\}$ & Non-arithmetic optimisation family \\
```

If Table I becomes too tall, split it into a compressibility-family table and
a circuit-workload table (this branch does not require the split, but it is
the natural place to do so when σ_quota lands).

---

## Workload paragraph

**Replace:**

```
For real-circuit grounding, we extract T-demand traces from three classes of Clifford$+T$ workloads: a ripple-carry adder, an integer multiplier, and an exact quantum Fourier transform (QFT) at the algorithmic level.
```

**With:**

```
We extract T-demand traces from six circuit workloads synthesised into
Clifford$+T$: a ripple-carry adder, an integer multiplier, a carry-lookahead
adder, a modular-arithmetic block, QAOA MaxCut circuits, and an exact quantum
Fourier transform (QFT) at the algorithmic level. The families are selected
to span serial arithmetic, parallel arithmetic, modular-arithmetic,
non-arithmetic optimisation, and Fourier-transform structure.

The carry-lookahead adder tests whether arithmetic implementation choices
affect bounded-delivery pressure for the same high-level operation. The
modular-arithmetic block is a controlled add-subtract-modulus chain motivated
by phase-estimation and order-finding workloads; it is included as a
structural surrogate for modular multiplication patterns rather than as a
verified mod-$N$ multiplier, and we report it as such throughout. The QAOA
MaxCut traces provide a non-arithmetic optimisation workload whose T-demand
structure depends on graph density and circuit depth. For QAOA, we evaluate
ring, random 3-regular, and dense Erd\H{o}s--R\'enyi graph instances, and
decompose the parameterised rotations using the same fixed Clifford$+T$
synthesis configuration as the other rotation-heavy traces.

For the workload families that combine many graph types, depths, and seeds,
we use a reduced bounded-delivery grid with $C \in \{1, 2, 3, 5, 7\}$ and
$B \in \{0, 4, 8, 12, 15\}$ (recorded as \texttt{grid\_type=reduced} in our
artefacts) to keep total Clifford$+T$ synthesis cost tractable. The other
circuit workloads and the compressibility-family evaluation use the full
$C \in 1..7$, $B \in 0..15$ grid (\texttt{grid\_type=full}). All
$\Delta_{\max}$ and slack-ratio values reported below use the static
scheduling policy $\sigma_{\mathrm{static}}$; per-instance result CSVs
released with this artefact also include $\sigma_{\mathrm{ca}}$ and
$\sigma_{\mathrm{smooth}}$ for the same grid.
```

---

## Section V-D — Real Workloads opening

**Replace the opening paragraph with:**

```
We next evaluate six circuit workloads: ripple-carry adder, integer
multiplier, carry-lookahead adder, modular-arithmetic block, QAOA MaxCut, and
exact QFT. The families span serial arithmetic, parallel arithmetic,
modular-arithmetic, non-arithmetic optimisation, and Fourier-transform
structure, and they test whether the structure--pressure--slowdown
relationship observed on the compressibility-family evaluation also appears
in workloads derived directly from algorithmic circuits.
```

---

## Before revised Fig. 6

```
Figure~\ref{fig:qft_vs_real} compares representative $n=8$ instances from each
family in structural--system space. The ripple-carry adder remains a
low-pressure workload, the integer multiplier exposes intermediate arithmetic
delivery pressure, the carry-lookahead adder tests whether a more parallel
implementation of addition changes the T-demand profile relative to
ripple-carry addition, the modular-arithmetic block reports the delivery
profile of the controlled add-subtract-modulus chain (a structural surrogate
for modular multiplication, not a verified mod-$N$ multiplier), the QAOA
MaxCut instance provides a non-arithmetic point whose delivery pressure
depends on graph density and circuit depth, and exact QFT continues to
occupy a high-pressure regime because Clifford$+T$ synthesis produces long
and dense T-demand traces. The QAOA pressure visible here is driven by the
Clifford$+T$ synthesis cost of the parameterised rotations in the cost and
mixer layers and not by reuse of the QFT trace.
```

**Caption:**

```
\caption{Representative $n=8$ circuit workloads in structural--system space.
The comparison spans serial arithmetic (Ripple, Multiplier), parallel
arithmetic (CLA), modular-arithmetic, non-arithmetic optimisation (QAOA), and
QFT. The left panel reports slack ratio; the right panel reports mean
$\Delta_{\max}$ on a symmetric logarithmic axis (zero values are annotated
explicitly). All values use the $\sigma_{\mathrm{static}}$ policy and average
$\Delta_{\max}$ over unique $C$ values. Ripple, CLA, Multiplier, modular-
arithmetic block, and exact QFT use the full bounded-delivery grid; the QAOA
bar uses the reduced grid (the dense $p=2$, seed-0 instance) -- see the
workload paragraph for grid definitions.}
```

---

## Before revised Fig. 7

```
Figure~\ref{fig:real_scaling} shows scaling behavior for the arithmetic and
modular-arithmetic workload families. Ripple-carry addition remains stable
across the evaluated range: slack ratio stays near~0.357 and mean
$\Delta_{\max}$ remains zero. The multiplier also stays in a low-slack
regime, but its delivery pressure grows gradually with $n$. The carry-
lookahead adder behaves differently: its slack ratio and mean $\Delta_{\max}$
both increase with problem size, showing that parallel-prefix arithmetic
creates more scheduling freedom but also stronger delivery pressure under a
depth-oriented schedule. The modular-arithmetic block remains close to the
ripple-carry profile because its construction is based on ripple-carry
building blocks.
```

## After revised Fig. 7

**Caption:**

```
\caption{Scaling behavior of arithmetic and modular-arithmetic workload
families under $\sigma_{\mathrm{static}}$. The left panel shows slack ratio
as problem size increases; the right panel shows mean $\Delta_{\max}$ using a
symmetric logarithmic scale, with zero-pressure workloads shown at zero.
Ripple-carry addition and the modular-arithmetic block remain delivery-
light, the multiplier shows gradually increasing delivery pressure, and the
Kogge--Stone carry-lookahead adder gains both structural flexibility and
delivery pressure with scale. QAOA is omitted from this scaling figure
because its rotation-synthesis-driven delivery pressure is shown separately
in Fig.~\ref{fig:qft_vs_real}.}
```

After Fig. 7 keep the QFT approximation paragraph; introduce it with:

```
We retain the QFT approximation study to illustrate that algorithmic or
decomposition-level changes can reduce delivery pressure even when ideal
dependency depth is unchanged.
```

---

## Compact representative table

Add this in Section V-D, immediately after the revised Fig. 6. The numbers come
from `barc_stage1/outputs/tables/real_trace_workload_families_representative_summary.csv`
(reduced-grid rows are flagged as such in that CSV; this table is meant to be
filled in by the script's output rather than typed manually). All numbers use
$\sigma_{\mathrm{static}}$.

```latex
\begin{table}[t]
\caption{Representative circuit-workload indicators under bounded delivery.
All values use the $\sigma_{\mathrm{static}}$ scheduling policy; the QAOA row
uses the reduced bounded-delivery grid, every other row uses the full grid.}
\label{tab:real_summary}
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccc}
\hline
Circuit workload & Instance & Slack & Mean $\Delta_{\max}$ & Slowdown rate \\
\hline
Ripple adder       & $n=8$            & ... & ... & ... \\
Multiplier         & $n=8$            & ... & ... & ... \\
CLA adder          & $n=8$            & ... & ... & ... \\
Mod. arith. block  & $n=8$            & ... & ... & ... \\
QAOA               & $n=8, p=2$, dense ER & ... & ... & ... \\
Exact QFT          & $n=8$            & ... & ... & ... \\
\hline
\end{tabular}
\end{table}
```

---

## Discussion — add at the end of the slack-ratio paragraph

```
The additional circuit workloads strengthen this compiler implication. The
contrast between ripple-carry and carry-lookahead addition tests whether
bounded-delivery pressure depends on implementation structure even for the
same arithmetic operation. The modular-arithmetic block (a structural
surrogate based on a controlled add-subtract-modulus chain) shows how
FTQC-relevant arithmetic blocks can amplify delivery pressure through
repeated controlled-modular-addition patterns, even when the block itself is
not a verified mod-$N$ multiplier. The QAOA MaxCut traces show that the same
diagnostic framework also applies outside arithmetic and QFT, where graph
density and circuit depth shape T-demand concentration; the high
$\Delta_{\max}$ visible for dense QAOA is driven by Clifford$+T$ synthesis of
parameterised rotations rather than by trace reuse from QFT.
```

---

## Limitations

**Replace:**

```
The constructed DAG families provide controlled variation in structural flexibility, but they do not substitute for exhaustive evaluation on large-scale real workloads.
```

**With:**

```
The compressibility-family evaluation provides controlled variation in
structural flexibility, and the arithmetic, modular-arithmetic, QAOA, and QFT
circuit workloads provide a broader workload spectrum. Nevertheless, these
workloads do not substitute for exhaustive evaluation on large-scale
fault-tolerant programs such as full phase-estimation, chemistry, and
simulation pipelines. The modular-arithmetic block is in particular a
structural surrogate based on a controlled add-subtract-modulus chain and is
not a verified mod-$N$ multiplier; it exposes the delivery pressure of a
modular-arithmetic block but should not be interpreted as an evaluation of
modular multiplication correctness.
```

---

## Conclusion — future-work sentence

**Replace with:**

```
Future work includes extending the framework to stochastic and spatial
delivery settings, evaluating it on larger end-to-end fault-tolerant programs
such as phase-estimation, chemistry, and simulation pipelines, developing
tighter bounds for persistent-backlog cases, and integrating these signals
into delivery-aware compiler passes and schedule-selection heuristics.
```

---

## Reduced-grid disclosure

Wherever the paper reports QAOA results, add a footnote or a sentence noting:

```
QAOA results use the reduced bounded-delivery grid $C \in \{1, 2, 3, 5, 7\}$,
$B \in \{0, 4, 8, 12, 15\}$. The reduced grid still spans low-, medium- and
high-capacity regimes and is recorded as \texttt{grid\_type=reduced} in the
artefacts.
```

---

## Synthesis-precision disclosure

Once, in the workload paragraph, add:

```
All Clifford$+T$ syntheses use \texttt{qiskit.transpile} with
\texttt{basis\_gates}~$=$~\texttt{[h, s, sdg, cx, t, tdg]},
\texttt{optimization\_level=1}, and \texttt{seed\_transpiler=42}, identical
to the existing QFT and adder/multiplier pipelines (recorded as
\texttt{synthesis\_precision} in every output row of the released artefacts).
Because QAOA contains parameterised rotations, all QAOA traces use the same
synthesis configuration, and reported T-counts and T-demand traces should be
interpreted relative to that configuration.
```

---

## Related Work — minimal patch

These are the only Related Work changes implied by this branch. Keep them
short and direct; do not add new subsections for QAOA, CLA, modular
arithmetic, or QFT.

**T-count / T-depth paragraph — add Clarino et al. 2025:**

```
Recent T-count and T-depth optimisation work, including Clarino et
al.~\cite{clarino2025tdepth}, continues to reduce the resource footprint of
fault-tolerant Clifford$+T$ programs at the algorithmic level. These results
are complementary to the present work: they reduce T-resource cost for a
given operation, while we ask how a fixed schedule's T-demand profile
interacts with bounded magic-state delivery.
```

**Magic-state delivery paragraph — add Silva et al. 2024 and Yamasaki \& Koashi 2024:**

```
On the supply side, recent magic-state delivery and distillation results,
including Silva et al.~\cite{silva2024magicstate} and Yamasaki and
Koashi~\cite{yamasaki2024magicstate}, characterise how distillation
throughput and supply variance constrain the rate at which magic states reach
logical qubits. These results focus on producing magic states at a desired
rate; the diagnostic developed in this paper focuses on the orthogonal
question of how a circuit's already-fixed schedule consumes that rate.
```

**Final diagnostic-gap paragraph — add as the last paragraph of Related Work:**

```
For compiler analysis, the missing piece is not another full
resource-constrained scheduler. It is a cheap diagnostic for fixed or
compiler-produced schedules: which circuits have enough structural freedom
to reshape T demand, and which schedules create delivery pressure under
bounded T-state supply? We address this gap with two quantities: slack ratio
for structural flexibility and $\Delta_{\max}$ for schedule-level
demand--supply imbalance.
```

---

## Appendix B cleanup (this branch)

The σ_quota / quota-respecting compiler probe is **not** promoted on this
branch. The follow-up branch `promote-sigma-quota` will handle that
promotion. On this branch, please keep Appendix B as-is in the manuscript;
do not add new σ_quota text and do not rename the existing appendix probe to
σ_quota.

If Appendix B is retained verbatim (recommended on this branch), no edit is
needed. If you choose to remove the appendix probe entirely, also remove or
revise the following references:

- `Appendix~\ref{app:compiler_probe}`
- `\sigma_{\text{quota}}` (do not introduce in this branch)
- "quota-respecting scheduling probe"

If positive-gap cases are removed from the appendix, also remove references
to:

- `Appendix~\ref{app:gap_cases}`
- `Fig.~\ref{fig:appendix_gap_cases}`

If positive-gap cases are retained, update wording only; do not regenerate
the figure.
