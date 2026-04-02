from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import FINAL_PAPER_DIR, TABLE_DIR, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    stochastic_df = pd.read_csv(TABLE_DIR / "stochastic_supply_ranking_summary.csv")
    routing_df = pd.read_csv(TABLE_DIR / "routing_proxy_ranking_summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

    axes[0].plot(
        stochastic_df["p_acc"],
        stochastic_df["spearman_delta_max_vs_mean_slowdown"],
        marker="o",
        color="#4c78a8",
        label="Nominal $\\Delta_{\\max}$",
    )
    axes[0].plot(
        stochastic_df["p_acc"],
        stochastic_df["spearman_expected_service_delta_max_vs_mean_slowdown"],
        marker="o",
        color="#f58518",
        label="Expected-service deficit",
    )
    axes[0].set_title("Stochastic Supply Sensitivity")
    axes[0].set_xlabel("$p_{\\mathrm{acc}}$")
    axes[0].set_ylabel("Spearman with mean slowdown")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        routing_df["routing_alpha"],
        routing_df["spearman_nominal_delta_max_vs_proxy_slowdown"],
        marker="o",
        color="#4c78a8",
        label="Nominal $\\Delta_{\\max}$",
    )
    axes[1].plot(
        routing_df["routing_alpha"],
        routing_df["spearman_proxy_delta_max_vs_proxy_slowdown"],
        marker="o",
        color="#f58518",
        label="Proxy-adjusted $\\Delta_{\\max}$",
    )
    axes[1].set_title("Routing Proxy Sensitivity")
    axes[1].set_xlabel("Routing proxy factor $\\alpha$")
    axes[1].set_ylabel("Spearman with slowdown")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    output_path = FINAL_PAPER_DIR / "robustness_stochastic_routing_summary_final.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    print(f"saved: {output_path}")
    print(f"saved: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
