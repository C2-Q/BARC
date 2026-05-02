# Workload-Family Extension Notes

This branch (`add-real-benchmark-families`) extends the workload spectrum with
three additional first-class circuit workloads:

- **CLA adder** (`cla_adder`) — parallel-prefix carry-lookahead-style adder.
- **Modular-arithmetic block** (`modular_multiplier` internal id) — controlled
  add-subtract-modulus chain. Honestly labelled as a structural surrogate, not
  a verified mod-$N$ multiplier; reported as **"Mod. arith. block"** in
  paper-facing text.
- **QAOA MaxCut** (`qaoa_maxcut`) — QAOA cost/mixer circuit on three graph
  types.

The existing ripple-carry adder, integer multiplier, and QFT families are
unchanged. The compressibility-family evaluation (constructed-DAG instances)
and the lower-bound validation on those instances are also unchanged on this
branch; the σ-quota promotion that modifies them is queued for a separate
`promote-sigma-quota` branch.

## Modular-arithmetic naming and correctness disclosure

The internal `workload_family` id is `modular_multiplier` for stability of the
output schema, but the paper-facing name is **"Mod. arith. block"** with the
role description **"Modular-arithmetic block; structural surrogate for modular
multiplication patterns"**. The implementation is a controlled
add-subtract-modulus chain (Qiskit `CDKMRippleCarryAdder`-based) and **does
not** implement the conditional add-back step on the sign bit that a verified
mod-$N$ multiplier requires. We do not claim mod-$N$ multiplication
correctness; the family is included to expose the delivery pressure of a
modular-arithmetic block, not to evaluate modular multiplication correctness.

LaTeX patches for the renaming are in `docs/paper_text_patches.md`.

## Scheduling policy set

The main policy set on this branch remains the original three:
`sigma_static` (`static_min`), `sigma_ca` (`capacity_aware_static`), and
`sigma_smooth` (`smoothed`). The `delivery_aware_slack` builder that lives in
`src/schedule.py` is exercised only by Appendix B's existing probe scripts on
this branch.

## Existing real-trace pipeline (re-used)

| Stage          | File                                                  |
| -------------- | ----------------------------------------------------- |
| Generators     | `src/real_trace/{adder,multiplier,qft}_trace.py`      |
| Clifford+T     | `src/real_trace/adder_trace.py::to_clifford_t`        |
| DAG / scheduler| `src/schedule.py`, `src/real_trace/circuit_to_dag.py` |
| Simulator      | `src/simulator.py`                                    |
| Metrics        | `src/metrics.py`                                      |
| Real-trace scan| `src/real_trace/analysis.py::evaluate_real_trace_policies` |
| Figure scripts | `scripts/run_real_trace_scaling.py`, `scripts/run_qft_real_trace_scan.py` |

## New generator modules

| Family                    | File                                                |
| ------------------------- | --------------------------------------------------- |
| CLA adder                 | `src/real_trace/cla_adder_trace.py`                 |
| Modular-arithmetic block  | `src/real_trace/modular_multiplier_trace.py`        |
| QAOA MaxCut               | `src/real_trace/qaoa_maxcut_trace.py`               |

Each module exports `generate_*_circuit` and `generate_and_save_*_trace`
mirroring the existing real-trace API, so downstream code (DAG conversion,
schedulers, simulator, metrics) is reused unchanged. The shared
`to_clifford_t` helper performs the Clifford+T basis transpilation (see
"Synthesis precision" below).

## Variant labels and constants

| Workload                    | `variant`                                  | Notes                                            |
| --------------------------- | ------------------------------------------ | ------------------------------------------------ |
| CLA adder                   | `kogge_stone_parallel_prefix`              | Out-of-place; ancillae not uncomputed.           |
| Modular-arithmetic block    | `controlled_add_subtract_modulus_chain`    | Structural surrogate; not a verified mod-$N$ multiplier. |
| QAOA MaxCut                 | (none; `graph_type` and `qaoa_p` instead)  | Fixed (γ, β) = (0.7/(k+1), 0.3/(k+1)).           |

For the modular-arithmetic block, the modulus is fixed to `2**n - 1` and the
constant to `3` for every size. The generator does not implement the
conditional add-back on the sign bit that a fully-verified mod-N multiplier
requires; it is labelled as a "controlled add-subtract-modulus chain" in
metadata to avoid overclaiming.

## QAOA configuration

- Sizes: `n in [6, 8, 10]`.
- Depths: `p in [1, 2]`.
- Graph types: `ring`, `random_3_regular`, `erdos_renyi_dense` (G(n, p_e=0.5)).
- Seeds: `[0]` for `ring` (deterministic); `[0, 1, 2]` for the two random families.
- Angles: `gamma_k = 0.7 / (k+1)`, `beta_k = 0.3 / (k+1)`. Not optimised.

## Delivery grid

| Family                       | `grid_type` | (C, B) values                                          |
| ---------------------------- | ----------- | ------------------------------------------------------ |
| Ripple adder                 | `full`      | `C in 1..7`, `B in 0..15` (paper_config.C_VALUES, B_VALUES) |
| Multiplier                   | `full`      | same                                                   |
| CLA adder                    | `full`      | same                                                   |
| Modular-arithmetic block     | `full`      | same                                                   |
| QAOA MaxCut                  | `reduced`   | `C in {1, 2, 3, 5, 7}`, `B in {0, 4, 8, 12, 15}`       |
| Exact QFT                    | `full`      | same                                                   |

The `grid_type` column is recorded in every output row so the two regimes are
never silently mixed. The reduced grid for QAOA is required because Clifford+T
synthesis of the parameterised rotations produces long traces (often >100k
cycles); the reduced grid still preserves low-, medium- and high-capacity
regimes.

## Synthesis precision

The Clifford+T synthesis is performed by `qiskit.transpile` with
`basis_gates=['h', 's', 'sdg', 'cx', 't', 'tdg']`, `optimization_level=1`, and
`seed_transpiler=42`, identical to the existing QFT and adder/multiplier
pipelines. There is no explicit ε parameter; we record the synthesis
configuration as the string

```
qiskit_transpile_optimization_level_1_seed_42_basis_h_s_sdg_cx_t_tdg
```

in every output row's `synthesis_precision` field. Because the QAOA cost and
mixer unitaries are parameterised rotations, this configuration determines
the resulting T-counts and T-demand traces — the QAOA traces should therefore
be interpreted relative to this synthesis configuration.

## Aggregation and policy used in the new tables

All summary CSVs (`real_trace_workload_families_representative_summary.csv`,
`real_trace_workload_families_scaling_summary.csv`, `qaoa_summary.csv`)
project onto the **σ_static** scheduling policy (`policy == "static_min"`)
and average Δ_max over **unique C values** in the grid (since static-schedule
Δ_max depends only on C, not on B). Per-instance result CSVs at
`outputs/real_trace/<instance_id>_results.csv` contain the full per-(C, B,
policy) grid for σ_static, σ_ca, and σ_smooth.

This matches the legacy real-trace pipeline:
`scripts/run_real_trace_scaling.py` and `scripts/run_qft_real_trace_scan.py`
also filter to σ_static and average over unique C values; the legacy
`delta_max_mean` column has the same definition as the new `mean_delta_max`
column.

## Legacy-workload consistency

The augmenter in `scripts/run_real_trace_workload_families.py` pulls
ripple-adder, multiplier, and exact-QFT rows from the existing
`outputs/tables/real_trace_scaling_summary.csv` and
`outputs/tables/qft_real_trace_summary.csv` summaries into the same schema.
We have verified that:

- The legacy summaries use the same σ_static policy filter and the same
  `drop_duplicates(subset=["C"])` aggregation (`scripts/run_real_trace_scaling.py:59-77`).
- The legacy summaries use the **full** delivery grid (`C in 1..7`, `B in
  0..15`), recorded in the augmented rows as `grid_type="full"`.
- The legacy synthesis basis is identical to the new families
  (`basis_gates=['h','s','sdg','cx','t','tdg']`, `optimization_level=1`,
  `seed_transpiler=42`); the augmenter fills in the `synthesis_precision`
  string accordingly.
- The legacy column `delta_max_mean` is renamed to the new convention
  `mean_delta_max` during the pull-in. No definitional change.
- The legacy variant labels (`ripple_carry`, `qiskit_multiplier_gate`,
  `synth_qft_full`) are preserved in the augmented rows.

No definitional drift between the legacy ripple-adder/multiplier/QFT rows and
the new CLA/modular-arithmetic/QAOA rows.

## QAOA Δ_max sanity check

For the representative QAOA instance
(`qaoa_maxcut_n8_p2_erdos_renyi_dense_s0`):

- Source trace file: `barc_stage1/data/real_traces/qaoa_maxcut_n8_p2_erdos_renyi_dense_s0.csv`
  (not the QFT trace path — no trace reuse).
- Trace length: 281,206 logical layers.
- T_count: 496,393 (sum of demand).
- T_depth / static depth: 281,206 (one layer per non-zero demand entry).
- peak_T_demand: 6 (cost-layer CNOT-RZ-CNOT and mixer-layer RX bursts of
  parallel parameterised rotations).
- First 10 nonzero demand entries (timestep, demand): (2,2), (3,2), (4,2),
  (6,2), (8,2), (10,2), (12,2), (13,2), (14,2), (15,2).
- Graph edges: 11.
- Synthesis basis: same Clifford+T configuration as adders/QFT (see above);
  parameterised rotations decompose into long T-rich sequences via Qiskit's
  rotation synthesis.

Δ_max (static_min) per C in the reduced grid:

```
C=1 → 215188
C=2 → 0
C=3 → 0
C=5 → 0
C=7 → 0
mean over unique C = 215188 / 5 = 43037.6
```

The high mean is a real artefact of Clifford+T synthesis of parameterised
rotations (22 RZ + 16 RX rotations for n=8, p=2, 11-edge graph; each rotation
synthesises into thousands of T gates, producing long bursts that exceed
delivery capacity at C=1). It is not caused by QFT trace reuse, duplicate
trace reuse, or aggregation bug. The raw trace file path, edge count, and
synthesis configuration are recorded in the per-instance result CSV.

## Repository-path mapping

The original specification used a `results/` directory; this repository
already stores generated tables under `barc_stage1/outputs/tables/` and
per-instance result rows under `barc_stage1/outputs/real_trace/`. To minimise
disruption, the extension follows the existing layout:

| Spec name                                                      | Actual path                                               |
| -------------------------------------------------------------- | --------------------------------------------------------- |
| `results/results_real_workloads_raw.csv`                       | `barc_stage1/outputs/tables/real_trace_workload_families_raw.csv` |
| `results/results_real_workloads_representative_summary.csv`    | `barc_stage1/outputs/tables/real_trace_workload_families_representative_summary.csv` |
| `results/results_real_workloads_scaling_summary.csv`           | `barc_stage1/outputs/tables/real_trace_workload_families_scaling_summary.csv` |
| `results/results_qaoa_summary.csv`                             | `barc_stage1/outputs/tables/qaoa_summary.csv`             |

## Figure path mapping (repo → LaTeX)

The repo paths are intentionally **not** changed to match the LaTeX paths.
The packaging step (`prepare_submission_bundle.py` /
`publish_paper_figures.py`) is responsible for the copy and rename:

| Repo path                                                   | LaTeX path                                                        |
| ----------------------------------------------------------- | ----------------------------------------------------------------- |
| `barc_stage1/outputs/figures/qft_vs_real_traces.pdf`        | `figures/final_paper/qft_vs_other_real_traces_final.pdf`          |
| `barc_stage1/outputs/figures/real_trace_scaling.pdf`        | `figures/final_paper/real_trace_scaling_final.pdf`                |

No `_final`-suffixed file names are introduced in this branch. The mapping
is also documented at the top of `docs/paper_text_patches.md`.

## Sizes successfully evaluated

All 47 instances evaluated successfully on the configured grid (no skips):

- CLA adder: `n ∈ {4, 8, 12, 16}` (4 instances, full grid).
- Modular-arithmetic block: `n ∈ {4, 6, 8}` with modulus `2**n - 1`,
  multiplier constant `3` (3 instances, full grid).
- QAOA MaxCut: `n ∈ {6, 8, 10}` × `p ∈ {1, 2}` × {ring (1 seed),
  random_3_regular (3 seeds), erdos_renyi_dense (3 seeds)} = 42 instances,
  reduced grid.

Sizes that exceed the `TRACE_LENGTH_LIMIT = 600_000` or `T_COUNT_LIMIT =
1_000_000` thresholds (inherited from the QFT pipeline) are kept as
trace-only rows with `evaluated=0` and a `skip_reason` field; on the
configured sizes there were none.

## Commands to reproduce

```
bash scripts/reproduce_real_workload_families.sh
```

This regenerates all workload-family CSVs and the two updated real-workload
figures without touching the constructed-DAG (compressibility-family) or
lower-bound experiments. It assumes that the existing
`barc_stage1/outputs/tables/family_summary.csv`,
`real_trace_scaling_summary.csv` and `qft_real_trace_summary.csv` are present;
the existing main pipeline produces them.

## What this branch does not change

- Constructed-DAG (compressibility-family) experiments.
- Lower-bound validation, the 4,904-finite-instance count, and
  `lower_bound_vs_actual_final.pdf`.
- `predictor_comparison.pdf` and `incremental_predictive_gain_final.pdf`.
- `qft_approximation_reduced_grid_final.pdf`.
- `robustness_stochastic_routing_summary_final.pdf` and
  `lower_bound_gap_cases_final.pdf`.
- The `delivery_aware_slack` policy stays inside Appendix B's probe; promotion
  to σ_quota will happen on the follow-up branch `promote-sigma-quota`.
