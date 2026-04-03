from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def build_figure() -> None:
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    t = np.linspace(0, 10, 500)
    C = 0.6
    B = 2.0

    ct = C * t
    ct_b = C * t + B
    a_sigma = 0.4 * t + 4.5 * norm.cdf(t, loc=5, scale=0.8)

    ax.plot(t, ct, color="black", linestyle="--", linewidth=1.5)
    ax.plot(t, ct_b, color="black", linestyle="-", linewidth=1.5)
    ax.plot(t, a_sigma, color="black", linestyle="-", linewidth=2.5)

    ax.text(9.2, C * 9.2 - 0.45, r"$Ct$", fontsize=14)
    ax.text(
        9.45,
        C * 9.45 + B + 0.12,
        r"$Ct+B$",
        fontsize=14,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
    )
    ax.text(8.1, a_sigma[-50] + 0.65, r"$A_\sigma(t)$", fontsize=14)

    peak_t = 5.3
    peak_a = 0.4 * peak_t + 4.5 * norm.cdf(peak_t, loc=5, scale=0.8)
    peak_ct = C * peak_t

    ax.annotate(
        "",
        xy=(peak_t, peak_ct),
        xytext=(peak_t, peak_a),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    ax.text(
        peak_t + 0.25,
        (peak_a + peak_ct) / 2 - 0.05,
        r"$\Delta_{\max}$",
        fontsize=14,
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
    )

    left_t = 4.75
    left_a = 0.4 * left_t + 4.5 * norm.cdf(left_t, loc=5, scale=0.8)
    left_ct_b = C * left_t + B
    ax.annotate(
        "",
        xy=(left_t, left_ct_b),
        xytext=(left_t, left_a),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    ax.text(
        left_t - 0.55,
        (left_a + left_ct_b) / 2 + 0.55,
        r"$\max(0,\Delta_{\max}-B)$",
        fontsize=12,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
    )

    ax.annotate(
        "",
        xy=(0, B),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="|-|", color="black", lw=1.4),
    )
    ax.text(-0.78, B / 2, "initial\nbuffer $B$", fontsize=12, va="center", ha="right")

    ax.text(3.9, 7.55, "stall / backlog peak", fontsize=12, style="italic")

    ax.set_xlabel(r"time step $t$", fontsize=14)
    ax.set_ylabel("cumulative T-state count", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "delta_max_illustration.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "delta_max_illustration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
