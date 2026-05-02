from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    generate_adder_circuit,
    generate_and_save_adder_trace,
    save_trace,
    to_clifford_t,
)
from src.real_trace.multiplier_trace import (
    generate_and_save_multiplier_trace,
    generate_multiplier_circuit,
)
from src.real_trace.qft_trace import (
    generate_and_save_qft_trace,
    generate_qft_circuit,
)
from src.real_trace.cla_adder_trace import (
    CLA_VARIANT,
    generate_and_save_cla_adder_trace,
    generate_cla_adder_circuit,
)
from src.real_trace.modular_multiplier_trace import (
    MODULAR_MULTIPLIER_VARIANT,
    generate_and_save_modular_multiplier_trace,
    generate_modular_multiplier_circuit,
)
from src.real_trace.qaoa_maxcut_trace import (
    QAOA_GRAPH_TYPES,
    build_maxcut_graph,
    build_qaoa_instance,
    default_qaoa_angles,
    generate_and_save_qaoa_maxcut_trace,
    generate_qaoa_maxcut_circuit,
)
from src.real_trace.analysis import (
    export_real_trace_threshold_summary,
    export_real_trace_workload_statistics,
    ensure_real_trace_exists,
    estimate_compressibility,
    evaluate_real_trace_policies,
    generate_real_trace_comparison,
    load_real_trace,
    load_real_trace_from_csv,
    load_real_trace_from_json,
    plot_adder_vs_multiplier_capacity_scan,
    plot_adder_vs_multiplier_trace,
    plot_cross_workload_critical_buffer,
    plot_cross_workload_critical_capacity,
    plot_cross_workload_improvement_gap,
    plot_real_trace_capacity_scan,
    plot_real_trace_demand,
    plot_real_trace_buffer_transition,
    summarize_real_trace,
)
from src.real_trace.circuit_slack import CircuitSlackMetrics, compute_circuit_slack_metrics
from src.real_trace.circuit_to_dag import quantum_circuit_to_internal_dag
from src.real_trace.stats_utils import build_real_trace_note

__all__ = [
    "ensure_real_trace_exists",
    "estimate_compressibility",
    "evaluate_real_trace_policies",
    "export_real_trace_threshold_summary",
    "export_real_trace_workload_statistics",
    "extract_t_demand_trace",
    "generate_adder_circuit",
    "generate_and_save_adder_trace",
    "generate_and_save_cla_adder_trace",
    "generate_and_save_modular_multiplier_trace",
    "generate_and_save_multiplier_trace",
    "generate_and_save_qaoa_maxcut_trace",
    "generate_and_save_qft_trace",
    "generate_cla_adder_circuit",
    "generate_modular_multiplier_circuit",
    "generate_multiplier_circuit",
    "generate_qaoa_maxcut_circuit",
    "generate_qft_circuit",
    "build_maxcut_graph",
    "build_qaoa_instance",
    "default_qaoa_angles",
    "CLA_VARIANT",
    "MODULAR_MULTIPLIER_VARIANT",
    "QAOA_GRAPH_TYPES",
    "generate_real_trace_comparison",
    "load_real_trace",
    "load_real_trace_from_csv",
    "load_real_trace_from_json",
    "plot_adder_vs_multiplier_capacity_scan",
    "plot_adder_vs_multiplier_trace",
    "plot_cross_workload_critical_buffer",
    "plot_cross_workload_critical_capacity",
    "plot_cross_workload_improvement_gap",
    "plot_real_trace_buffer_transition",
    "plot_real_trace_capacity_scan",
    "plot_real_trace_demand",
    "save_trace",
    "summarize_real_trace",
    "to_clifford_t",
    "build_real_trace_note",
    "CircuitSlackMetrics",
    "compute_circuit_slack_metrics",
    "quantum_circuit_to_internal_dag",
]
