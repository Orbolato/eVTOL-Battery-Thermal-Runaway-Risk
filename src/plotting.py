"""Programmatic regeneration of every manuscript figure (instructions Sec. 9.6).

All figures are produced only from arrays computed in this run (no
manual editing) and saved as vector PDF for print quality. Avoids
decorative titles, ambiguous "HIGH/LOW RISK" zone labels without units,
and mixed-unit thresholds presented without qualification.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})


def fig_distribution(ptr: np.ndarray, p_base: float, outfile: str) -> None:
    """Distribution of the dimensionless risk multiplier R = P_TR / P_base (defined in
    Sec. III-E). Plotting R rather than absolute P_TR keeps the only external scaling
    choice (the assumed baseline reference, R=1 by construction) visible on the axis
    without implying P_TR itself is a verified per-charge probability. Deliberately
    does NOT plot the EASA per-flight-hour threshold: the two are different exposure
    metrics (Sec. III-C), and co-plotting them invites exactly the unit-conflation the
    manuscript text argues against. The EASA value is discussed in prose only."""
    r = ptr / p_base
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.logspace(np.log10(max(r.min(), 1e-3)), np.log10(r.max()), 80)
    ax.hist(r, bins=bins, color="#4C72B0", alpha=0.85, label="Simulated $R$ (independent baseline model)")
    ax.set_xscale("log")
    ax.axvline(1.0, color="#DD8452", linestyle="--", label="Assumed baseline reference ($R=1$)")
    ax.set_xlabel("$R$ (dimensionless risk multiplier)")
    ax.set_ylabel("Simulated count")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


VARIABLE_DISPLAY_LABELS = {
    "SOC0": "$SOC_0$",
    "N_cycles": "$N_{cycles}$",
    "T_bat": "$T_{bat}$",
    "T_amb": "$T_{amb}$",
    "C_rate": "C-rate",
    "R_int0": "$R_{int,0}$",
    "sigma_V": "$\\sigma_V$",
}


def fig_sensitivity(rows: list[dict], outfile: str) -> None:
    names = [VARIABLE_DISPLAY_LABELS.get(r["variable"], r["variable"]) for r in rows]
    rhos = [r["rho_s"] for r in rows]
    lo = [r["rho_s"] - r["ci95_lo"] for r in rows]
    hi = [r["ci95_hi"] - r["rho_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    y = np.arange(len(names))
    ax.barh(y, rhos, xerr=[lo, hi], color="#4C72B0", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman $\\rho_s$ with risk multiplier $R$ (bootstrap 95% CI, sub-sampled)")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def fig_sobol(rows: list[dict], outfile: str) -> None:
    names = [r["variable"] for r in rows]
    s1 = [max(r["S1"], 0) for r in rows]
    st = [max(r["ST"], 0) for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(y - 0.2, s1, height=0.4, label="First-order $S_1$", color="#4C72B0")
    ax.barh(y + 0.2, st, height=0.4, label="Total-order $S_T$", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index (independent baseline model)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(soc_edges: np.ndarray, crate_edges: np.ndarray, mean_grid: np.ndarray,
                 p_base: float, outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.pcolormesh(crate_edges, soc_edges, mean_grid / p_base, shading="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean $R$ (dimensionless)")
    ax.set_xlabel("C-rate (C)")
    ax.set_ylabel("SOC at charge initiation (%)")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def fig_convergence(rows: list[dict], outfile: str) -> None:
    ns = [r["N"] for r in rows]
    means = [r["mean_P_TR"] for r in rows]
    hw = [r["ci95_halfwidth"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(ns, means, yerr=hw, marker="o", capsize=3, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo sample size $N$")
    ax.set_ylabel("Mean $P_{TR}$ (95% CI of the mean)")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def fig_dependency_graph(outfile: str) -> None:
    """Diagram of structural dependence / double-counting pathways (no external
    graph library dependency). Node coordinates are chosen so that the two
    root-to-P_TR pathways form a symmetric, non-crossing "fan" layout (the
    cycle-count pathway confined to y>=0.5, the C-rate pathway to y<=0.5,
    meeting only at P_TR), and arrows are shrunk at both ends (shrinkA/shrinkB,
    matched to the node marker radius) so they terminate at the node's edge
    rather than piercing into the marker and overlapping its label. The
    legend text lives in the figure caption, not inside the plot, to avoid
    overlapping the diagram; axis limits are padded beyond the node extremes
    so the marker circles are not clipped at the figure margins."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axis("off")
    nodes = {
        "N_cycles": (0.03, 0.82),
        "f_cycles": (0.36, 0.92),
        "R_int": (0.36, 0.68),
        "f_deg": (0.70, 0.60),
        "C_rate": (0.03, 0.18),
        "f_Crate": (0.36, 0.08),
        "T_onset": (0.36, 0.32),
        "f_thermal": (0.70, 0.40),
        "P_TR": (1.00, 0.50),
    }
    edges = [
        ("N_cycles", "f_cycles"), ("N_cycles", "R_int"), ("R_int", "f_deg"),
        ("C_rate", "f_Crate"), ("C_rate", "T_onset"), ("T_onset", "f_thermal"),
        ("f_cycles", "P_TR"), ("f_deg", "P_TR"), ("f_Crate", "P_TR"), ("f_thermal", "P_TR"),
    ]
    labels = {
        "N_cycles": "$N_{cycles}$",
        "f_cycles": "$f_{cycles}$",
        "R_int": "$R_{int}$",
        "f_deg": "$f_{deg}$",
        "C_rate": "$C_{rate}$",
        "f_Crate": "$f_{Crate}$",
        "T_onset": "$T_{onset}$",
        "f_thermal": "$f_{thermal}$",
        "P_TR": "$P_{TR}$",
    }
    marker_pt_radius = 15  # points; matches marker_size below via s = (2*r)**2
    marker_size = (2 * marker_pt_radius) ** 2
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2,
                                     shrinkA=marker_pt_radius + 2, shrinkB=marker_pt_radius + 2),
                    annotation_clip=False)
    for name, (x, y) in nodes.items():
        # Every intermediate node mediating a pathway (i.e. every node other than the
        # two root inputs and the P_TR sink) is colored orange, matching the caption's
        # "orange nodes are the intermediate factors mediating each pathway" -- not just
        # the subset individually isolated by the structural ablation.
        is_root_or_sink = name in ("N_cycles", "C_rate", "P_TR")
        color = "#4C72B0" if is_root_or_sink else "#DD8452"
        ax.scatter([x], [y], s=marker_size, color=color, alpha=0.25, zorder=2, clip_on=False)
        ax.text(x, y, labels[name], ha="center", va="center", fontsize=8, zorder=3, clip_on=False)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(outfile, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


STRATEGY_DISPLAY_LABELS = {
    "soc_85": "SOC ceiling: 85%",
    "soc_80": "SOC ceiling: 80%",
    "crate_5": "C-rate ceiling: 5C",
    "crate_4": "C-rate ceiling: 4C",
    "cycles_1200": "Replace at 1200 cycles",
    "combined_moderate": "Combined—moderate",
    "combined_aggressive": "Combined—aggressive",
}


def fig_mitigation_pareto(rows: list[dict], outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for r in rows:
        if r["strategy"] == "baseline":
            continue
        label = STRATEGY_DISPLAY_LABELS.get(r["strategy"], r["strategy"])
        ax.scatter(r["risk_reduction_pct"], r["turnaround_time_index"], s=60, color="#4C72B0")
        ax.annotate(label, (r["risk_reduction_pct"], r["turnaround_time_index"]),
                    fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Risk reduction vs. baseline (%)")
    ax.set_ylabel("Turnaround-time index (rel. to baseline)")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
