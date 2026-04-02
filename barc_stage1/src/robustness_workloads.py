from __future__ import annotations

from dataclasses import dataclass
from src.dag import generate_dag
from src.paper_config import FAMILIES, REPRESENTATIVE_SEED
from src.real_trace.analysis import load_real_trace_from_csv, smooth_trace
from src.schedule import build_schedule
from src.trace import schedule_to_trace_bundle
from src.utils import OUTPUT_ROOT


SYNTHETIC_FAMILY_SPECS: dict[str, dict[str, float | int]] = {
    "high_compressibility": {"num_layers": 8, "width": 8, "t_ratio": 0.45},
    "medium_compressibility": {"num_layers": 8, "width": 7, "t_ratio": 0.45},
    "low_compressibility": {"num_layers": 7, "width": 6, "t_ratio": 0.45},
}

REAL_TRACE_DIR = OUTPUT_ROOT.parent / "data" / "real_traces"


@dataclass(frozen=True)
class WorkloadVariant:
    workload_scope: str
    workload_name: str
    policy: str
    demand: list[int]

    @property
    def variant_id(self) -> str:
        return f"{self.workload_name}:{self.policy}"

    @property
    def trace_length(self) -> int:
        return len(self.demand)

    @property
    def total_t(self) -> int:
        return int(sum(self.demand))

    @property
    def peak_demand(self) -> int:
        return max(self.demand, default=0)


def build_synthetic_workload_variants(policies: tuple[str, ...] = ("static_min", "smoothed")) -> list[WorkloadVariant]:
    variants: list[WorkloadVariant] = []
    for family in FAMILIES:
        spec = SYNTHETIC_FAMILY_SPECS[family]
        dag = generate_dag(
            family=family,
            num_layers=int(spec["num_layers"]),
            width=int(spec["width"]),
            t_ratio=float(spec["t_ratio"]),
            seed=REPRESENTATIVE_SEED,
        )
        for policy in policies:
            schedule = build_schedule(dag, policy=policy)
            trace_bundle = schedule_to_trace_bundle(dag, schedule)
            variants.append(
                WorkloadVariant(
                    workload_scope="synthetic",
                    workload_name=f"{family}_seed{REPRESENTATIVE_SEED}",
                    policy=policy,
                    demand=list(trace_bundle.demand),
                )
            )
    return variants


def build_real_workload_variants(
    trace_names: tuple[str, ...] = ("adder_n8", "multiplier_n8", "qft_n8"),
    policies: tuple[str, ...] = ("static_min", "smoothed"),
) -> list[WorkloadVariant]:
    variants: list[WorkloadVariant] = []
    for trace_name in trace_names:
        trace_path = REAL_TRACE_DIR / f"{trace_name}.csv"
        trace = load_real_trace_from_csv(str(trace_path))
        for policy in policies:
            demand = list(trace) if policy == "static_min" else smooth_trace(trace)
            variants.append(
                WorkloadVariant(
                    workload_scope="real",
                    workload_name=trace_name,
                    policy=policy,
                    demand=demand,
                )
            )
    return variants
