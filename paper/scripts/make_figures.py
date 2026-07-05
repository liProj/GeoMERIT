# -*- coding: utf-8 -*-
"""Unified, journal-style figure set for the GeoMERIT BDCC revision.

Reads only existing result/figure CSV data (plus public FORCE curve extracts for
the two case-study wells) and regenerates all paper figures with one consistent
style: same fonts, same lithology palette, panel labels, vector PDF + 600 dpi PNG.

Run:  python make_figures.py
Outputs to ../figures (revision_v3/figures).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]          # essay/
FIG_SRC = ROOT / "figures"                          # original figure CSV data
RES = ROOT / "results"
DATA = Path(__file__).resolve().parents[1] / "data" # well curve extracts
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Global style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Liberation Sans",   # Arial-metric-compatible
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})

FULL_W = 6.89   # 17.5 cm MDPI full width
HALF_W = 3.35   # 8.5 cm

# semantic lithology palette, identical in every figure ------------------------
LITHO_ORDER = ["Shale", "Sandstone", "Sandstone/Shale", "Limestone", "Marl",
               "Tuff", "Halite", "Chalk", "Coal", "Dolomite", "Anhydrite", "Basement"]
LITHO_COLORS = {
    "Shale": "#8C8C8C",            # grey
    "Sandstone": "#F4C542",        # yellow
    "Sandstone/Shale": "#C9A66B",  # tan
    "Limestone": "#4C9FC4",        # blue family ----
    "Chalk": "#9DD0E8",
    "Marl": "#2E7E8F",
    "Dolomite": "#8E5BA6",         # purple
    "Halite": "#F59B5C",           # red-orange family ----
    "Anhydrite": "#D94F3D",
    "Coal": "#1A1A1A",             # black
    "Tuff": "#7CB75C",             # green (volcanic)
    "Basement": "#6B3A2A",         # dark brown
}
LITHO_CMAP = ListedColormap([LITHO_COLORS[n] for n in LITHO_ORDER])

# geological group order used by penalty + confusion matrices ------------------
GROUP_ORDER = ["Shale", "Sandstone", "Sandstone/Shale",
               "Limestone", "Chalk", "Marl", "Dolomite",
               "Halite", "Anhydrite", "Coal", "Tuff", "Basement"]
GROUPS = [("clastic", 0, 3), ("carbonate–marl", 3, 7), ("evaporite", 7, 9),
          ("", 9, 10), ("", 10, 11), ("", 11, 12)]

# method colors ----------------------------------------------------------------
C_BASE = "#9AA3AD"
C_RFE = "#3D6FA8"
C_GEO = "#C8503C"
C_COMPLETE = "#3D6FA8"
C_DEFICIENT = "#E08214"

TAIL_SET = ["Tuff", "Halite", "Coal", "Dolomite", "Anhydrite", "Basement"]
REF = {"wf1": 0.7272, "bf1": 0.4100, "pen": -0.6289}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print("saved", name)


def panel_label(ax, s, dx=-0.08, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


# ============================================================================
# Figure 1 — dataset overview (a) map (b) class support (c) missingness matrix
# ============================================================================

def fig1():
    wl = pd.read_csv(FIG_SRC / "fig01_well_locations.csv")
    dist = pd.read_csv(FIG_SRC / "fig02_lithology_distribution.csv")
    mh = pd.read_csv(FIG_SRC / "fig03_missingness_heatmap.csv")

    curves = [c for c in mh.columns if c not in ("well_id", "x_loc", "y_loc", "z_loc")]
    M = mh[curves].values

    fig = plt.figure(figsize=(FULL_W, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.35], wspace=0.42,
                          left=0.06, right=0.92, bottom=0.22, top=0.90)
    norm = Normalize(0, 1)
    cmap = plt.cm.YlGnBu

    # (a) map -----------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = (wl.x_loc - wl.x_loc.min()) / 1000.0
    y = (wl.y_loc - wl.y_loc.min()) / 1000.0
    ax.scatter(x, y, c=wl.mean_missing_rate, cmap=cmap, norm=norm, s=16,
               edgecolor="k", linewidth=0.25, zorder=3)
    for wid, marker in [("16/7-6", "*"), ("35/9-7", "*")]:
        row = wl[wl.well_id == wid]
        if len(row):
            xi = float((row.x_loc - wl.x_loc.min()) / 1000.0)
            yi = float((row.y_loc - wl.y_loc.min()) / 1000.0)
            ax.scatter([xi], [yi], marker=marker, s=110, facecolor="none",
                       edgecolor=C_GEO, linewidth=1.2, zorder=4)
            ax.annotate(wid, (xi, yi), textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color=C_GEO, fontweight="bold")
    ax.set_xlabel("Easting offset (km)")
    ax.set_ylabel("Northing offset (km)")
    ax.set_title("Well locations (118 wells)", fontsize=9)
    panel_label(ax, "(a)")

    # (b) class support ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    d = dist.sort_values("count", ascending=True)
    cols = [LITHO_COLORS[n] for n in d.class_name]
    bars = ax.barh(np.arange(len(d)), d["count"], color=cols, edgecolor="k",
                   linewidth=0.3, height=0.72)
    for i, (n, c) in enumerate(zip(d.class_name, d["count"])):
        lab = f"{c/1000:.0f}k" if c >= 10000 else f"{c:,}"
        ax.text(c * 1.25, i, lab, va="center", fontsize=7)
        if n in TAIL_SET:
            bars[i].set_hatch("////")
    ylabels = [n + " †" if n in TAIL_SET else n for n in d.class_name]
    ax.set_yticks(np.arange(len(d)), ylabels, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(80, 6e6)
    ax.set_xlabel("Labeled samples (log scale)")
    tail_pct = dist[dist.class_name.isin(TAIL_SET)].percent.sum()
    ax.set_title("Long-tail label support", fontsize=9)
    ax.text(0.97, 0.04, f"† tail classes = {tail_pct:.1f}% of rows",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color="#555555")
    panel_label(ax, "(b)")

    # (c) missingness heatmap ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(len(curves)), curves, rotation=90, fontsize=6)
    step = 10
    ax.set_yticks(np.arange(0, len(mh), step),
                  [mh.well_id.iloc[i] for i in range(0, len(mh), step)], fontsize=6)
    ax.set_xlabel("Log curve")
    ax.set_ylabel("Well")
    ax.set_title("Structured curve-suite missingness", fontsize=9)
    ax.spines.top.set_visible(True); ax.spines.right.set_visible(True)
    panel_label(ax, "(c)")

    cax = fig.add_axes([0.935, 0.24, 0.013, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Missing fraction", fontsize=8)
    save(fig, "fig1_dataset_overview")


# ============================================================================
# Figure 2 — penalty matrix (group-ordered, annotated, Blues like confusion)
# ============================================================================

def matrix_in_group_order(df):
    df = df.set_index("true_class")
    return df.loc[GROUP_ORDER, GROUP_ORDER]


def draw_group_frames(ax, lw=1.4, color="#444444", label_inside=True):
    for name, a, b in GROUPS:
        if not name:
            continue
        r = patches.Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False,
                              edgecolor=color, linewidth=lw, zorder=5)
        ax.add_patch(r)
        if label_inside:
            ax.text(b - 0.42, a - 0.30, name, ha="right", va="top",
                    fontsize=6.4, color=color, style="italic", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))


def fig2():
    P = matrix_in_group_order(pd.read_csv(FIG_SRC / "fig06_penalty_matrix.csv"))
    V = P.values.astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.45))
    ax.imshow(V, cmap="Blues", vmin=0, vmax=4)
    for i in range(12):
        for j in range(12):
            if i == j:
                ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor="#F2F2F2", edgecolor="none"))
                ax.text(j, i, "0", ha="center", va="center", fontsize=6.4, color="#999999")
            else:
                v = V[i, j]
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=6.4,
                        color="white" if v > 2.6 else "#1d3a5f")
    draw_group_frames(ax)
    ax.set_xticks(range(12), GROUP_ORDER, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(12), GROUP_ORDER, fontsize=7.5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.spines.top.set_visible(True); ax.spines.right.set_visible(True)
    ax.set_title("FORCE 2020 geological penalty matrix (argmax decoding ignores it)",
                 fontsize=9)
    save(fig, "fig2_penalty_matrix")


# ============================================================================
# Figure 3 — framework: challenge -> layer -> mechanism -> verified by
# ============================================================================

LAYERS = [
    ("Data pathology", "Layer", "Mechanism", "Verified by", None),
    ("Non-random curve-suite\nmissingness; cross-well\nheterogeneity",
     "Representation\nlayer  (§4.1)",
     "missing/outlier flags + sentinels;\nwithin-well robust z-score;\nmulti-scale windows; curve-suite\nfingerprint; strat./spatial context",
     "RQ3 (Fig. 6)\nRQ5 (Fig. 8)", "#3D6FA8"),
    ("Long-tailed labels;\nambiguous lithology\nboundaries",
     "Learning\nlayer  (§4.2)",
     "effective-number class weight ×\nboundary weight ×\nconfidence weight",
     "RQ2 (Tab. 4)\nRQ4 (Fig. 7)", "#E08214"),
    ("Single-backbone bias;\nrare-class signal diluted\nin 12-way objective",
     "Posterior\nlayer  (§4.3)",
     "LightGBM + XGBoost + CatBoost\ngeometric fusion; coarse-to-fine;\none-vs-rest tail experts",
     "RQ2 (Tab. 5)\nRQ4 (Fig. 7)", "#2E8B57"),
    ("Asymmetric geological\nmisclassification cost",
     "Decision\nlayer  (§4.4)",
     "class-prior logit adjustment (τ);\ntail gates; Bayes-risk decoding\nover FORCE penalty matrix",
     "RQ1 (Fig. 4–5)\nRQ6 (Fig. 9)", "#8E5BA6"),
]


def fig3():
    fig, ax = plt.subplots(figsize=(FULL_W, 4.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    colx = [(0.1, 2.3), (2.75, 4.45), (4.9, 8.1), (8.5, 9.95)]
    rowy = [(7.45, 9.45), (5.05, 7.05), (2.65, 4.65), (0.25, 2.25)]
    heads = LAYERS[0]
    for ci, (x0, x1) in enumerate(colx):
        ax.text((x0 + x1) / 2, 9.85, heads[ci], ha="center", va="center",
                fontsize=9, fontweight="bold")
    for ri, (chal, layer, mech, rq, col) in enumerate(LAYERS[1:]):
        y0, y1 = rowy[ri]
        cells = [chal, layer, mech, rq]
        for ci, (x0, x1) in enumerate(colx):
            face = "#FFFFFF" if ci != 1 else col
            tcol = "white" if ci == 1 else "#222222"
            box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                 boxstyle="round,pad=0.02,rounding_size=0.12",
                                 linewidth=1.1, edgecolor=col, facecolor=face)
            ax.add_patch(box)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, cells[ci], ha="center", va="center",
                    fontsize=7 if ci in (0, 2) else 7.6, color=tcol,
                    fontweight="bold" if ci == 1 else "normal", linespacing=1.25)
            if ci < 3:
                ax.annotate("", xy=(colx[ci + 1][0] - 0.03, (y0 + y1) / 2),
                            xytext=(x1 + 0.03, (y0 + y1) / 2),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1))
        if ri < 3:
            xm = (colx[1][0] + colx[1][1]) / 2
            ax.annotate("", xy=(xm, rowy[ri + 1][1] + 0.05), xytext=(xm, y0 - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))
    save(fig, "fig3_framework")


# ============================================================================
# Figure 4 — main results dumbbell with gain annotations
# ============================================================================

MAIN = {
    "Weighted F1": (0.6640, 0.7272, 0.7526, "+3.5%"),
    "Boundary F1": (0.3510, 0.4100, 0.4567, "+11.4%"),
    "Penalty": (-0.7800, -0.6289, -0.5886, "+6.4%"),
}


def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.95))
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.98, top=0.78, bottom=0.30)
    for ax, (name, (b, r, g, gain)) in zip(axes, MAIN.items()):
        ax.plot([b, r], [0, 0], color="#CCCCCC", lw=2, zorder=1)
        ax.plot([r, g], [0, 0], color=C_GEO, lw=2.6, zorder=2)
        ax.scatter([b], [0], s=46, color=C_BASE, zorder=3, label="Baseline XGBoost")
        ax.scatter([r], [0], s=52, color=C_RFE, zorder=3, label="RFE + XGBoost")
        ax.scatter([g], [0], s=86, color=C_GEO, marker="D", zorder=4, label="GeoMERIT")
        span0 = abs(g - b)
        ax.text(b, -0.16, f"{b:.4f}", ha="center", va="top", fontsize=7.4, color=C_BASE)
        ax.text(r - 0.012 * span0, -0.16, f"{r:.4f}", ha="right", va="top",
                fontsize=7.4, color=C_RFE)
        ax.text(g + 0.012 * span0, -0.16, f"{g:.4f}", ha="left", va="top",
                fontsize=7.4, color=C_GEO)
        ax.annotate(gain, xy=((r + g) / 2, 0.085), ha="center", fontsize=10,
                    color=C_GEO, fontweight="bold")
        span = abs(g - b)
        ax.set_xlim(min(b, g) - 0.16 * span, max(b, g) + 0.22 * span)
        ax.set_ylim(-0.5, 0.42)
        ax.set_yticks([])
        ax.spines.left.set_visible(False)
        ax.set_title(name, fontsize=9.5)
        if name == "Penalty":
            ax.annotate("closer to 0 is better →", xy=(0.97, 0.86),
                        xycoords="axes fraction", ha="right", fontsize=7.4,
                        color="#555555", style="italic")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    save(fig, "fig4_main_results")


# ============================================================================
# Figure 5 — per-fold forest plot vs RFE reference
# ============================================================================

def fig5():
    pf = pd.read_csv(RES / "perfold_geomerit.csv")
    meta = [("wf1", "Weighted F1", "7/10 folds > ref.", "p = 0.16"),
            ("bf1", "Boundary F1", "9/10 folds > ref.", "p = 0.0039"),
            ("pen", "Penalty", "6/10 folds > ref.", "p = 0.28")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.75), sharey=True)
    fig.subplots_adjust(wspace=0.14, left=0.06, right=0.99, top=0.86, bottom=0.26)
    for ax, (m, title, ann, pv) in zip(axes, meta):
        v = pf[m].values
        ref = REF[m]
        folds = np.arange(10)
        above = v > ref
        ax.axvline(ref, color="#888888", ls="--", lw=1.1, zorder=1)
        ax.scatter(v[above], folds[above], s=34, color=C_GEO, zorder=3)
        ax.scatter(v[~above], folds[~above], s=34, facecolor="white",
                   edgecolor=C_GEO, linewidth=1.1, zorder=3)
        for f in folds:
            ax.plot([ref, v[f]], [f, f], color=C_GEO, alpha=0.25, lw=0.9, zorder=2)
        mean, sd = v.mean(), v.std(ddof=1)
        ax.errorbar([mean], [10.4], xerr=[[sd], [sd]], fmt="D", color=C_GEO,
                    ms=8, capsize=3, lw=1.3, zorder=4)
        ax.text(0.03, 0.965, f"{ann}\nWilcoxon {pv}", transform=ax.transAxes,
                fontsize=7.6, va="top",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#BBBBBB", lw=0.7))
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(-1.9, 11.3)
        ax.invert_yaxis()
    axes[0].set_yticks(list(range(10)) + [10.4],
                       [f"fold {i}" for i in range(10)] + ["mean ± sd"], fontsize=7.5)
    fig.text(0.53, 0.075, "Metric value on the held-out wells of each fold"
             "   (dashed line = pooled RFE+XGBoost reference)",
             ha="center", fontsize=8.6)
    save(fig, "fig5_perfold_forest")


# ============================================================================
# Figure 6 — robustness as missingness-threshold curves
# ============================================================================

def fig6():
    rb = pd.read_csv(FIG_SRC / "fig09_robustness_complete_vs_deficient.csv")
    thr = ["0.1/0.4", "0.2/0.5", "0.3/0.6"]
    meta = [("weighted_f1", "Weighted F1", REF["wf1"]),
            ("boundary_f1", "Boundary F1", REF["bf1"]),
            ("penalty", "Penalty", REF["pen"])]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45))
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.99, top=0.86, bottom=0.32)
    xs = np.arange(3)
    for ax, (m, title, ref) in zip(axes, meta):
        ax.axhline(ref, color="#888888", ls="--", lw=1.0, zorder=1)
        ax.text(2.02, ref, "RFE full-fleet ref.", fontsize=6.8, color="#666666",
                va="bottom", ha="right")
        for grp, color, ls, mk, mfc, lab in [
                ("feature_complete", C_COMPLETE, "-", "o", C_COMPLETE, "feature-complete wells"),
                ("feature_deficient", C_DEFICIENT, "--", "o", "white", "feature-deficient wells")]:
            sub = rb[rb.group == grp].set_index("threshold").reindex(thr)
            yv = sub[m].values.astype(float)
            nw = sub["wells"].values
            mask = ~np.isnan(yv) & (nw > 0)
            ax.plot(xs[mask], yv[mask], ls, color=color, marker=mk, ms=5,
                    mfc=mfc, mec=color, label=lab, zorder=3)
            for xi, yi, wi in zip(xs[mask], yv[mask], nw[mask]):
                ax.annotate(f"{int(wi)}w", (xi, yi), textcoords="offset points",
                            xytext=(0, -11), ha="center", fontsize=6.2, color=color)
        ax.set_xticks(xs, thr)
        ax.set_xlabel("Missingness split threshold")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-0.3, 2.3)
    axes[0].set_ylabel("Metric value")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig6_missingness_robustness")


# ============================================================================
# Figure 7 — class-wise diagnosis composite (a) lollipop (b) confusion (c) tail
# ============================================================================

def fig7():
    pc = pd.read_csv(FIG_SRC / "fig08_per_class_metrics.csv")
    cm = matrix_in_group_order(pd.read_csv(FIG_SRC / "fig08_confusion_matrix_normalized.csv"))
    tail = pd.read_csv(FIG_SRC / "fig_tail_f1.csv")

    fig = plt.figure(figsize=(FULL_W, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 0.85], width_ratios=[0.92, 1.08],
                          hspace=0.42, wspace=0.34, left=0.13, right=0.97,
                          top=0.95, bottom=0.085)

    # (a) support-aware F1 lollipop --------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    pcg = pc.set_index("name").loc[GROUP_ORDER].reset_index()
    ypos = np.arange(len(pcg))[::-1]
    sizes = 14 + 26 * (np.log10(pcg["support"]) - 2)
    for yi, (_, row) in zip(ypos, pcg.iterrows()):
        col = LITHO_COLORS[row["name"]]
        ax.plot([0, row.f1], [yi, yi], color=col, lw=1.3, alpha=0.85, zorder=2)
        ax.scatter([row.f1], [yi], s=max(sizes[_], 12), color=col,
                   edgecolor="k", linewidth=0.4, zorder=3)
        ax.text(row.f1 + 0.035, yi, f"{row.f1:.2f}", va="center", fontsize=7)
    ax.axvline(0.5, color="#BBBBBB", ls=":", lw=0.9, zorder=1)
    ax.text(0.5, -0.62, "F1 = 0.5", fontsize=6.6, color="#888888",
            ha="center", va="top")
    labels = [f"{n}  ({s/1000:.0f}k)" if s >= 10000 else f"{n}  ({s:,})"
              for n, s in zip(pcg["name"], pcg["support"])]
    ax.set_yticks(ypos, labels, fontsize=7.5)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.7, len(pcg) - 0.3)
    ax.set_xlabel("Per-class F1 (marker size ~ log support)")
    panel_label(ax, "(a)", dx=-0.42)

    # (b) row-normalized confusion with group frames ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    V = cm.values.astype(float)
    ax.imshow(V, cmap="Blues", vmin=0, vmax=1)
    for i in range(12):
        for j in range(12):
            if V[i, j] >= 0.02:
                ax.text(j, i, f"{V[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=5.6, color="white" if V[i, j] > 0.55 else "#1d3a5f")
    draw_group_frames(ax, lw=1.5, color="#C8503C", label_inside=False)
    ax.annotate("clastic\ncluster", xy=(2.55, 1.0), xytext=(5.0, 0.6), fontsize=6.8,
                color="#C8503C", va="center",
                arrowprops=dict(arrowstyle="->", color="#C8503C", lw=0.9))
    ax.annotate("carbonate–marl\ncluster", xy=(6.55, 5.0), xytext=(8.6, 3.6),
                fontsize=6.8, color="#C8503C", va="center",
                arrowprops=dict(arrowstyle="->", color="#C8503C", lw=0.9))
    ax.set_xticks(range(12), GROUP_ORDER, rotation=45, ha="right", fontsize=6.8)
    ax.set_yticks(range(12), GROUP_ORDER, fontsize=6.8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.spines.top.set_visible(True); ax.spines.right.set_visible(True)
    panel_label(ax, "(b)", dx=-0.30)

    # (c) tail panel -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    t = tail.copy()
    ok = t.f1 >= 0.65
    cols = ["#2E8B57" if o else "#C0392B" for o in ok]
    bars = ax.bar(np.arange(len(t)), t.f1, color=cols, width=0.62,
                  edgecolor="k", linewidth=0.4)
    ax.axhline(0.65, color="#555555", ls="--", lw=1.0)
    ax.text(5.45, 0.67, "recovery goal 0.65", fontsize=7, color="#555555", ha="right")
    for xi, (f, o) in enumerate(zip(t.f1, ok)):
        ax.text(xi, f + 0.025, f"{f:.2f}", ha="center", fontsize=7.6,
                color="#2E8B57" if o else "#C0392B", fontweight="bold")
    ax.set_xticks(np.arange(len(t)), t.class_name, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Tail-class F1")
    ax.set_title("4/6 tail classes recovered above 0.65; Dolomite and Basement remain failure modes",
                 fontsize=8.6)
    leg = [patches.Patch(fc="#2E8B57", label="recovered (F1 ≥ 0.65)"),
           patches.Patch(fc="#C0392B", label="failure mode")]
    ax.legend(handles=leg, frameon=False, loc="upper right", fontsize=7.4)
    panel_label(ax, "(c)", dx=-0.055)
    save(fig, "fig7_class_diagnosis")


# ============================================================================
# Figure 8 — two held-out well tracks with curves + lithology strips
# ============================================================================

SUITE = ["GR", "SGR", "RHOB", "DRHO", "NPHI", "PEF", "DTC", "DTS", "RDEP", "RSHA",
         "RMED", "RXO", "RMIC", "SP", "CALI", "BS", "DCAL", "ROP", "ROPA", "MUDWEIGHT"]
CURVE_STYLE = {  # track curve, color, xlabel, log?
    "GR": ("#2E8B57", "GR (gAPI)", False),
    "RHOB": ("#C0392B", "RHOB (g/cm³)", False),
    "NPHI": ("#3D6FA8", "NPHI (v/v)", False),
    "DTC": ("#8E5BA6", "DTC (µs/ft)", False),
    "RDEP": ("#333333", "RDEP (Ω·m)", True),
}


def litho_strip(ax, depth, cls, title):
    img = cls.values.reshape(-1, 1)
    ax.imshow(img, aspect="auto", cmap=LITHO_CMAP, vmin=-0.5, vmax=11.5,
              extent=[0, 1, depth.max(), depth.min()], interpolation="nearest")
    ax.set_xticks([])
    ax.set_title(title, fontsize=7.2)


def well_row(fig, gs_row, well, track_csv, label, ann=None):
    tr = pd.read_csv(FIG_SRC / track_csv)
    cur = pd.read_csv(DATA / f"wellfull_{well.replace('/', '_')}.csv")
    cur = cur.rename(columns={"DEPTH_MD": "depth"})
    df = pd.merge_asof(tr.sort_values("depth"), cur.sort_values("depth"),
                       on="depth", direction="nearest", tolerance=0.08)
    miss = df[SUITE].isna().mean(axis=1)
    acc = (df.y_true == df.y_pred).mean()
    n_absent = int(df[SUITE].isna().all(axis=0).sum())
    absent = [c for c in SUITE if df[c].isna().all()]

    axes = [fig.add_subplot(gs_row[k]) for k in range(9)]
    d = df.depth
    for k, (cname, (col, xlab, lg)) in enumerate(CURVE_STYLE.items()):
        ax = axes[k]
        v = df[cname]
        ax.plot(v, d, color=col, lw=0.55)
        if lg:
            ax.set_xscale("log")
        # shade contiguous gaps > 2 m
        nan = v.isna().values
        if nan.any() and not nan.all():
            idx = np.where(nan)[0]
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for s in splits:
                d0, d1 = d.iloc[s[0]], d.iloc[s[-1]]
                if d1 - d0 > 2:
                    ax.axhspan(d0, d1, color="#BBBBBB", alpha=0.45, hatch="////", lw=0)
        if nan.all():
            ax.text(0.5, 0.5, f"{cname}\nabsent", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#888888")
        ax.set_title(xlab, fontsize=6.6)
        ax.tick_params(labelsize=6)
        if not lg:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
        ax.set_ylim(d.max(), d.min())
        if k > 0:
            ax.set_yticklabels([])
    # missing-rate track
    ax = axes[5]
    ax.fill_betweenx(d, 0, miss, color="#999999", alpha=0.7, lw=0)
    ax.set_xlim(0, 1)
    ax.set_title("Suite missing\nfraction", fontsize=6.6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(d.max(), d.min())
    ax.set_yticklabels([])
    if absent:
        lines = [", ".join(absent[i:i + 3]) for i in range(0, len(absent), 3)]
        ax.text(0.5, 0.012, f"{n_absent}/20 curves absent:\n" + ",\n".join(lines),
                transform=ax.transAxes, fontsize=4.8, ha="center", va="bottom",
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
    # lithology strips
    litho_strip(axes[6], d, df.y_true, "True")
    litho_strip(axes[7], d, df.y_pred, "Predicted")
    ax = axes[8]
    hit = (df.y_true == df.y_pred).astype(int).values.reshape(-1, 1)
    ax.imshow(hit, aspect="auto", cmap=ListedColormap(["#C0392B", "#2E8B57"]),
              vmin=0, vmax=1, extent=[0, 1, d.max(), d.min()], interpolation="nearest")
    ax.set_xticks([])
    ax.set_title("Hit", fontsize=7.2)
    for ax in axes[6:]:
        ax.set_yticklabels([])
        ax.tick_params(labelsize=6)
        ax.set_ylim(d.max(), d.min())
    axes[0].set_ylabel("Depth MD (m)", fontsize=8)
    axes[0].text(-0.62, 1.075, label, transform=axes[0].transAxes, fontsize=10,
                 fontweight="bold")
    axes[0].text(0.0, 1.075, f"Well {well}  —  point accuracy {acc*100:.1f}%",
                 transform=axes[0].transAxes, fontsize=8.6)
    if ann:
        a0, a1, txt = ann
        for ax in axes[6:9]:
            ax.axhline(a0, color="k", lw=0.7, ls=":")
            ax.axhline(a1, color="k", lw=0.7, ls=":")
        axes[8].annotate(txt, xy=(1.15, (a0 + a1) / 2), xycoords=("axes fraction", "data"),
                         fontsize=6.4, rotation=90, va="center", ha="left", color="#333333")
    return df


def fig8():
    fig = plt.figure(figsize=(FULL_W, 8.6))
    gs = fig.add_gridspec(2, 9, hspace=0.34, wspace=0.30,
                          left=0.075, right=0.965, top=0.93, bottom=0.115)
    well_row(fig, [gs[0, k] for k in range(9)], "16/7-6",
             "fig10_complete_well_track.csv", "(a)",
             ann=(2200, 2400, "carbonate interval"))
    well_row(fig, [gs[1, k] for k in range(9)], "35/9-7",
             "fig11_deficient_well_track.csv", "(b)")
    handles = [patches.Patch(fc=LITHO_COLORS[n], label=n) for n in LITHO_ORDER]
    handles += [patches.Patch(fc="#2E8B57", label="correct"),
                patches.Patch(fc="#C0392B", label="incorrect")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.018))
    save(fig, "fig8_well_tracks")


# ============================================================================
# Figure 9 — decision-layer sensitivity (tau sweep + Dolomite gate Pareto)
# ============================================================================

def pareto_front(x, y):
    order = np.argsort(x)[::-1]
    xs, ys, best = [], [], -np.inf
    for i in order:
        if y[i] > best:
            xs.append(x[i]); ys.append(y[i]); best = y[i]
    return np.array(xs[::-1]), np.array(ys[::-1])


def fig9():
    g = pd.read_csv(FIG_SRC / "fig_decode_grid_results.csv")
    tau = g[g.stage == "tau"].sort_values("tau")
    dol = pd.read_csv(FIG_SRC / "fig_dolomite_gate_tradeoff.csv")
    adopted = g[(g.stage == "global_tail") & (g.tau == 0.2) &
                (g.dol_gamma == 8.0) & (g.dol_theta == 0.05)]
    if len(adopted) == 0:
        adopted = dol[(dol.dol_gamma == 8.0) & (dol.dol_theta == 0.05)]
        adopted = adopted[adopted.tau == adopted.tau.max()]
    arow = adopted.iloc[-1]
    print("adopted point:", arow[["stage", "tau", "dol_gamma", "dol_theta",
                                  "dolomite_f1", "penalty"]].to_dict())

    fig = plt.figure(figsize=(FULL_W, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.12], height_ratios=[1.45, 1],
                          hspace=0.12, wspace=0.30, left=0.075, right=0.985,
                          top=0.875, bottom=0.13)

    # stable region: tau where weighted_f1 within 0.002 of its max
    stable = tau[tau.weighted_f1 >= tau.weighted_f1.max() - 0.002].tau
    s0, s1 = stable.min(), stable.max()

    ax1 = fig.add_subplot(gs[0, 0])
    for m, lab, col in [("weighted_f1", "Weighted F1", "#3D6FA8"),
                        ("macro_f1", "Macro F1", "#2E8B57"),
                        ("boundary_f1", "Boundary F1", "#E08214"),
                        ("tail_mean_f1", "Tail mean F1", "#8E5BA6")]:
        ax1.plot(tau.tau, tau[m], color=col, label=lab, lw=1.2)
        yv = float(tau.loc[(tau.tau - 0.2).abs().idxmin(), m])
        ax1.scatter([0.2], [yv], color=col, s=22, zorder=4)
    ax1.axvspan(s0, s1, color="#F2E8C9", alpha=0.6, zorder=0)
    ax1.axvline(0.2, color="#444444", ls="--", lw=1.0)
    ax1.text((s0 + s1) / 2, 0.985, "stable region", transform=
             matplotlib.transforms.blended_transform_factory(ax1.transData, ax1.transAxes),
             fontsize=6.8, color="#8a7530", ha="center", va="top", style="italic")
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("F1 metrics")
    ax1.legend(frameon=False, fontsize=7, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.005), columnspacing=0.9, handlelength=1.3)
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(tau.tau, tau.penalty, color="#333333", lw=1.3)
    yv = float(tau.loc[(tau.tau - 0.2).abs().idxmin(), "penalty"])
    ax2.scatter([0.2], [yv], color="#333333", s=24, zorder=4)
    ax2.axvspan(s0, s1, color="#F2E8C9", alpha=0.6, zorder=0)
    ax2.axvline(0.2, color="#444444", ls="--", lw=1.0)
    ax2.annotate("τ = 0.2 adopted", xy=(0.2, yv), xytext=(0.135, yv - 0.0035),
                 fontsize=7.4, color="#444444", ha="right",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax2.set_xlabel("Logit-adjustment temperature τ")
    ax2.set_ylabel("Penalty\n(higher is better)")

    ax3 = fig.add_subplot(gs[:, 1])
    sc = ax3.scatter(dol.dolomite_f1, dol.penalty, c=dol.dol_gamma, cmap="viridis",
                     s=15, alpha=0.75, edgecolor="none")
    px, py = pareto_front(dol.dolomite_f1.values, dol.penalty.values)
    ax3.plot(px, py, color="#C8503C", lw=1.4, label="Pareto frontier", zorder=3)
    ax3.scatter([arow.dolomite_f1], [arow.penalty], marker="*", s=240, color="#C8503C",
                edgecolor="k", linewidth=0.6, zorder=5)
    ax3.annotate("adopted (\u03b3=8.0, \u03b8=0.05)",
                 xy=(arow.dolomite_f1, arow.penalty),
                 xytext=(arow.dolomite_f1 - 0.012, arow.penalty - 0.014),
                 fontsize=7.6, ha="right",
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#C8503C"))
    cb = fig.colorbar(sc, ax=ax3, pad=0.015)
    cb.set_label("Dolomite gate \u03b3", fontsize=8)
    ax3.set_xlabel("Dolomite F1")
    ax3.set_ylabel("Penalty (higher is better)")
    ax3.legend(frameon=False, fontsize=7.4, loc="lower left")
    panel_label(ax3, "(b)")
    save(fig, "fig9_decision_layer")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9()
    print("all figures written to", OUT)
