from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomerit.labels import LABEL_NAMES, TAIL_IDS, load_penalty_matrix
from geomerit.metrics import boundary_f1, macro_f1, penalty_score, tail_f1, weighted_f1


FIG_DPI = 220
COLORS = {
    "baseline": "#6B7280",
    "rfe": "#2563EB",
    "geomerit": "#DC2626",
    "accent": "#059669",
    "tail": "#7C3AED",
    "warn": "#D97706",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="runs/remote_full10_fullfeat_3gbdt_c2f_tail")
    parser.add_argument("--feature_table", default=r"E:\geomerit_work\data\feature_table.parquet")
    parser.add_argument("--penalty", default="configs/penalty_matrix.csv")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = resolve(args.run_dir, project_root)
    feature_table = resolve(args.feature_table, project_root)
    penalty_path = resolve(args.penalty, project_root)
    out_dir = resolve(args.out_dir, project_root) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((run_dir / "decode_report.json").read_text(encoding="utf-8"))
    pred_df = pd.read_csv(run_dir / "decode_report.csv")
    pred_df["well_id"] = pred_df["well_id"].astype(str)
    y_true = pred_df["y_true"].to_numpy(int)
    y_pred = pred_df["y_pred"].to_numpy(int)
    well_id = pred_df["well_id"].to_numpy(str)
    penalty = load_penalty_matrix(penalty_path)

    meta = json.loads(feature_table.with_suffix(".meta.json").read_text(encoding="utf-8"))
    missing_cols = [c for c in meta["numeric"] if c.endswith("_is_missing")]
    curve_names = [c[: -len("_is_missing")] for c in missing_cols]
    feature_cols = list(
        dict.fromkeys(
            ["well_id", "DEPTH_MD", "label_idx", "missing_rate", "x_loc", "y_loc"]
            + missing_cols
        )
    )
    feature_df = pd.read_parquet(feature_table, columns=feature_cols)
    feature_df["well_id"] = feature_df["well_id"].astype(str)
    labeled_features = feature_df[feature_df["label_idx"] >= 0].copy()

    make_label_distribution(labeled_features, out_dir)
    make_well_location(labeled_features, out_dir)
    make_missingness_heatmap(labeled_features, missing_cols, curve_names, out_dir)
    make_penalty_heatmap(penalty, out_dir)
    make_main_metric_comparison(report, out_dir)
    make_reference_ablation(report, out_dir)
    make_tree_model_reference(report, out_dir)
    make_per_class_metrics(report, out_dir)
    make_tail_metrics(report, out_dir)
    make_confusion_matrix(y_true, y_pred, out_dir)
    make_robustness(pred_df, labeled_features, penalty, out_dir)
    make_decode_grid(run_dir, out_dir)
    make_well_tracks(pred_df, labeled_features, out_dir)

    manifest = sorted(str(p.relative_to(out_dir)) for p in out_dir.glob("*"))
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Wrote figures and data to {out_dir}")
    print(f"Files: {len(manifest)}")


def make_label_distribution(feature_df: pd.DataFrame, out_dir: Path) -> None:
    counts = feature_df["label_idx"].value_counts().reindex(range(12), fill_value=0).astype(int)
    data = pd.DataFrame(
        {
            "class_id": range(12),
            "class_name": LABEL_NAMES,
            "count": counts.to_numpy(),
            "percent": counts.to_numpy() / counts.sum() * 100.0,
        }
    )
    save_csv(data, out_dir / "fig02_lithology_distribution.csv")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = ["#475569" if i not in TAIL_IDS else COLORS["tail"] for i in range(12)]
    ax.bar(data["class_name"], data["count"], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Labeled samples (log scale)")
    ax.set_title("FORCE 2020 Lithology Class Distribution")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    add_bar_labels(ax, data["count"], fmt="{:.0f}", fontsize=7)
    save_figure(fig, out_dir / "fig02_lithology_distribution")


def make_well_location(feature_df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["x_loc", "y_loc", "missing_rate", "label_idx"]
    df = feature_df[["well_id"] + cols].copy()
    for col in ["x_loc", "y_loc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= -900, col] = np.nan
    well = (
        df.groupby("well_id")
        .agg(
            x_loc=("x_loc", "mean"),
            y_loc=("y_loc", "mean"),
            mean_missing_rate=("missing_rate", "mean"),
            labeled_rows=("label_idx", "size"),
        )
        .reset_index()
    )
    save_csv(well, out_dir / "fig01_well_locations.csv")
    plot_df = well.dropna(subset=["x_loc", "y_loc"])
    fig, ax = plt.subplots(figsize=(7, 7))
    if len(plot_df):
        sizes = 20 + 130 * (plot_df["labeled_rows"] / plot_df["labeled_rows"].max())
        sc = ax.scatter(
            plot_df["x_loc"],
            plot_df["y_loc"],
            s=sizes,
            c=plot_df["mean_missing_rate"],
            cmap="viridis",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.82)
        cb.set_label("Mean missing rate")
    ax.set_title("Well Location Overview")
    ax.set_xlabel("x_loc")
    ax.set_ylabel("y_loc")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out_dir / "fig01_well_locations")


def make_missingness_heatmap(
    feature_df: pd.DataFrame, missing_cols: list[str], curve_names: list[str], out_dir: Path
) -> None:
    heat = feature_df.groupby("well_id")[missing_cols].mean()
    heat.columns = curve_names
    heat["__mean__"] = heat.mean(axis=1)
    heat = heat.sort_values("__mean__", ascending=False).drop(columns="__mean__")
    save_csv(heat.reset_index(), out_dir / "fig03_missingness_heatmap.csv")

    fig_h = max(7.5, min(24, 0.12 * len(heat) + 2.5))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax.set_title("Missingness Rate by Well and Curve")
    ax.set_xlabel("Curve")
    ax.set_ylabel("Well (sorted by mean missingness)")
    ax.set_xticks(np.arange(len(curve_names)))
    ax.set_xticklabels(curve_names, rotation=45, ha="right", fontsize=7)
    step = max(1, len(heat) // 35)
    yticks = np.arange(0, len(heat), step)
    ax.set_yticks(yticks)
    ax.set_yticklabels(heat.index.to_numpy()[yticks], fontsize=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.72)
    cb.set_label("Missing rate")
    save_figure(fig, out_dir / "fig03_missingness_heatmap")


def make_penalty_heatmap(penalty: np.ndarray, out_dir: Path) -> None:
    data = pd.DataFrame(penalty, index=LABEL_NAMES, columns=LABEL_NAMES)
    save_csv(data.reset_index(names="true_class"), out_dir / "fig06_penalty_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(penalty, cmap="YlOrRd")
    ax.set_title("FORCE 2020 Penalty Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(LABEL_NAMES, fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Penalty cost")
    save_figure(fig, out_dir / "fig06_penalty_matrix")


def make_main_metric_comparison(report: dict, out_dir: Path) -> None:
    data = pd.DataFrame(
        [
            {
                "method": "Baseline XGBoost",
                "weighted_f1": 0.664,
                "boundary_f1": 0.351,
                "penalty": -0.780,
                "source": "RFE paper/reference in notes",
            },
            {
                "method": "RFE + XGBoost",
                "weighted_f1": 0.727,
                "boundary_f1": 0.410,
                "penalty": -0.628,
                "source": "RFE paper/reference in notes",
            },
            {
                "method": "GeoMERIT",
                "weighted_f1": report["weighted_f1"],
                "boundary_f1": report["boundary_f1"],
                "penalty": report["penalty"],
                "source": "This experiment",
            },
        ]
    )
    save_csv(data, out_dir / "fig07_main_metric_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [("weighted_f1", "Weighted F1"), ("boundary_f1", "Boundary F1"), ("penalty", "Penalty score")]
    colors = [COLORS["baseline"], COLORS["rfe"], COLORS["geomerit"]]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(data["method"], data[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(data[col]):
            ax.text(i, v + (0.015 if v >= 0 else -0.035), f"{v:.3f}", ha="center", fontsize=8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    fig.suptitle("Main Metric Comparison")
    save_figure(fig, out_dir / "fig07_main_metric_comparison")


def make_reference_ablation(report: dict, out_dir: Path) -> None:
    data = pd.DataFrame(
        [
            ["Baseline XGBoost", False, False, 0.6635, -0.7804, "RFE paper"],
            ["Baseline + Weighting", False, True, 0.7230, -0.6346, "RFE paper"],
            ["Baseline + FE", True, False, 0.7259, -0.6327, "RFE paper"],
            ["RFE + XGBoost", True, True, 0.7272, -0.6289, "RFE paper"],
            ["GeoMERIT", True, True, report["weighted_f1"], report["penalty"], "This experiment"],
        ],
        columns=["method", "feature_engineering", "weighting", "weighted_f1", "penalty", "source"],
    )
    save_csv(data, out_dir / "table02_reference_ablation_plus_geomerit.csv")
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(data))
    ax1.bar(x - 0.18, data["weighted_f1"], width=0.36, color=COLORS["rfe"], label="Weighted F1")
    ax1.set_ylabel("Weighted F1")
    ax1.set_ylim(0.62, max(0.78, data["weighted_f1"].max() + 0.02))
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, data["penalty"], width=0.36, color=COLORS["warn"], label="Penalty")
    ax2.set_ylabel("Penalty score")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["method"], rotation=25, ha="right")
    ax1.set_title("RFE Ablation Reference + GeoMERIT")
    ax1.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir / "table02_reference_ablation_plus_geomerit")


def make_tree_model_reference(report: dict, out_dir: Path) -> None:
    data = pd.DataFrame(
        [
            ["RF", 0.6918, 0.3840, -0.6776, "RFE paper"],
            ["RFE + RF", 0.7051, 0.3920, -0.6467, "RFE paper"],
            ["CatBoost", 0.6501, 0.3510, -0.7895, "RFE paper"],
            ["RFE + CatBoost", 0.6866, 0.3725, -0.7156, "RFE paper"],
            ["LightGBM", 0.6550, 0.3768, -0.8080, "RFE paper"],
            ["RFE + LightGBM", 0.7092, 0.4147, -0.6691, "RFE paper"],
            ["GeoMERIT ensemble", report["weighted_f1"], report["boundary_f1"], report["penalty"], "This experiment"],
        ],
        columns=["method", "weighted_f1", "boundary_f1", "penalty", "source"],
    )
    save_csv(data, out_dir / "table04_tree_model_reference_plus_geomerit.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, col, title in zip(
        axes,
        ["weighted_f1", "boundary_f1", "penalty"],
        ["Weighted F1", "Boundary F1", "Penalty score"],
    ):
        colors = [COLORS["geomerit"] if "GeoMERIT" in m else COLORS["rfe"] if "RFE" in m else COLORS["baseline"] for m in data["method"]]
        ax.barh(data["method"], data[col], color=colors)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Tree Model Generalization Reference + GeoMERIT")
    save_figure(fig, out_dir / "table04_tree_model_reference_plus_geomerit")


def make_per_class_metrics(report: dict, out_dir: Path) -> None:
    rows = []
    for i in range(12):
        item = report["per_class"][str(i)]
        rows.append({"class_id": i, **item})
    data = pd.DataFrame(rows)
    save_csv(data, out_dir / "fig08_per_class_metrics.csv")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(data))
    w = 0.24
    ax.bar(x - w, data["precision"], width=w, label="Precision", color="#0F766E")
    ax.bar(x, data["recall"], width=w, label="Recall", color="#2563EB")
    ax.bar(x + w, data["f1"], width=w, label="F1", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels(data["name"], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("GeoMERIT Per-Class Metrics")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir / "fig08_per_class_metrics")

    support = data[["class_id", "name", "support"]]
    save_csv(support, out_dir / "table01_lithology_support.csv")


def make_tail_metrics(report: dict, out_dir: Path) -> None:
    rows = []
    for cls in TAIL_IDS:
        rows.append(
            {
                "class_id": cls,
                "class_name": LABEL_NAMES[cls],
                "f1": float(report["tail_f1"][str(cls)]),
            }
        )
    data = pd.DataFrame(rows)
    save_csv(data, out_dir / "fig_tail_f1.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [COLORS["tail"] if cls != 9 else COLORS["warn"] for cls in data["class_id"]]
    ax.bar(data["class_name"], data["f1"], color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Tail-Class F1")
    ax.tick_params(axis="x", rotation=25)
    add_bar_labels(ax, data["f1"], fmt="{:.3f}", fontsize=8)
    save_figure(fig, out_dir / "fig_tail_f1")


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(12)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    data = pd.DataFrame(cm_norm, index=LABEL_NAMES, columns=LABEL_NAMES)
    save_csv(data.reset_index(names="true_class"), out_dir / "fig08_confusion_matrix_normalized.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("GeoMERIT Confusion Matrix (Row-Normalized)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(LABEL_NAMES, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75)
    save_figure(fig, out_dir / "fig08_confusion_matrix_normalized")


def make_robustness(pred_df: pd.DataFrame, feature_df: pd.DataFrame, penalty: np.ndarray, out_dir: Path) -> None:
    well_missing = feature_df.groupby("well_id")["missing_rate"].mean().rename("well_missing_rate").reset_index()
    df = pred_df.merge(well_missing, on="well_id", how="left")
    thresholds = [(0.1, 0.4), (0.2, 0.5), (0.3, 0.6)]
    rows = []
    for low, high in thresholds:
        for group_name, mask in [
            ("feature_complete", df["well_missing_rate"] <= low),
            ("feature_deficient", df["well_missing_rate"] >= high),
        ]:
            part = df[mask]
            if part.empty:
                row = {
                    "threshold": f"{low}/{high}",
                    "group": group_name,
                    "rows": 0,
                    "wells": 0,
                    "weighted_f1": np.nan,
                    "boundary_f1": np.nan,
                    "macro_f1": np.nan,
                    "penalty": np.nan,
                    "tail_mean_f1": np.nan,
                }
            else:
                yt = part["y_true"].to_numpy(int)
                yp = part["y_pred"].to_numpy(int)
                wid = part["well_id"].to_numpy(str)
                _, tm = tail_f1(yt, yp)
                row = {
                    "threshold": f"{low}/{high}",
                    "group": group_name,
                    "rows": len(part),
                    "wells": part["well_id"].nunique(),
                    "weighted_f1": weighted_f1(yt, yp),
                    "boundary_f1": boundary_f1(yt, yp, wid),
                    "macro_f1": macro_f1(yt, yp),
                    "penalty": penalty_score(yt, yp, penalty),
                    "tail_mean_f1": tm,
                }
            rows.append(row)
    data = pd.DataFrame(rows)
    save_csv(data, out_dir / "fig09_robustness_complete_vs_deficient.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, title in zip(
        axes,
        ["weighted_f1", "boundary_f1", "penalty"],
        ["Weighted F1", "Boundary F1", "Penalty score"],
    ):
        pivot = data.pivot(index="threshold", columns="group", values=metric)
        pivot.plot(kind="bar", ax=ax, color=[COLORS["accent"], COLORS["warn"]])
        ax.set_title(title)
        ax.set_xlabel("Complete/deficient threshold")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("GeoMERIT Robustness by Well Missingness")
    save_figure(fig, out_dir / "fig09_robustness_complete_vs_deficient")


def make_decode_grid(run_dir: Path, out_dir: Path) -> None:
    grid_path = run_dir / "decode_grid_results.json"
    if not grid_path.exists():
        return
    grid = pd.DataFrame(json.loads(grid_path.read_text(encoding="utf-8")))
    save_csv(grid, out_dir / "fig_decode_grid_results.csv")

    tau = grid[grid["stage"] == "tau"].sort_values("tau")
    if not tau.empty:
        fig, ax1 = plt.subplots(figsize=(8.5, 4.7))
        for col, color in [
            ("weighted_f1", COLORS["rfe"]),
            ("macro_f1", COLORS["accent"]),
            ("boundary_f1", COLORS["geomerit"]),
            ("tail_mean_f1", COLORS["tail"]),
        ]:
            ax1.plot(tau["tau"], tau[col], marker="o", label=col, color=color)
        ax1.set_xlabel("tau")
        ax1.set_ylabel("F1")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(tau["tau"], tau["penalty"], marker="s", linestyle="--", color=COLORS["warn"], label="penalty")
        ax2.set_ylabel("Penalty score")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=8, ncol=2)
        ax1.set_title("Decode Tau Trade-Off")
        save_figure(fig, out_dir / "fig_decode_tau_tradeoff")

    dol = grid[grid["stage"] == "dolomite"].copy()
    if not dol.empty:
        save_csv(dol, out_dir / "fig_dolomite_gate_tradeoff.csv")
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        sc = ax.scatter(
            dol["dolomite_f1"],
            dol["penalty"],
            c=dol["dol_gamma"],
            s=30 + 180 * dol["tau"].astype(float),
            cmap="plasma",
            alpha=0.75,
        )
        ax.set_xlabel("Dolomite F1")
        ax.set_ylabel("Penalty score")
        ax.set_title("Dolomite Gate: Tail Gain vs Penalty")
        ax.grid(True, alpha=0.25)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("dol_gamma")
        save_figure(fig, out_dir / "fig_dolomite_gate_tradeoff")


def make_well_tracks(pred_df: pd.DataFrame, feature_df: pd.DataFrame, out_dir: Path) -> None:
    row_missing = feature_df[["well_id", "DEPTH_MD", "missing_rate"]].copy()
    row_missing.rename(columns={"DEPTH_MD": "depth"}, inplace=True)
    merged = pred_df.merge(row_missing, on=["well_id", "depth"], how="left")
    well_stats = (
        merged.groupby("well_id")
        .agg(mean_missing=("missing_rate", "mean"), rows=("y_true", "size"))
        .query("rows >= 1000")
        .sort_values("mean_missing")
    )
    if well_stats.empty:
        return
    complete_well = well_stats.index[0]
    deficient_well = well_stats.index[-1]
    for tag, well in [("fig10_complete_well_track", complete_well), ("fig11_deficient_well_track", deficient_well)]:
        part = merged[merged["well_id"] == well].sort_values("depth").copy()
        save_csv(part, out_dir / f"{tag}.csv")
        plot_well_track(part, out_dir / tag, f"{well} (mean missing={part['missing_rate'].mean():.2f})")


def plot_well_track(df: pd.DataFrame, out_base: Path, title: str) -> None:
    step = max(1, len(df) // 6000)
    plot = df.iloc[::step].copy()
    depth = plot["depth"].to_numpy()
    cmap = plt.get_cmap("tab20", 12)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 8.5), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, col, name in [(axes[0], "y_true", "True"), (axes[1], "y_pred", "Pred")]:
        ax.scatter(np.zeros(len(plot)), depth, c=plot[col], cmap=cmap, vmin=-0.5, vmax=11.5, s=2)
        ax.set_title(name)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.18)
    axes[2].plot(plot["missing_rate"], depth, color=COLORS["warn"], linewidth=0.8)
    axes[2].set_title("Missing rate")
    axes[2].set_xlim(0, 1)
    axes[2].grid(True, alpha=0.25)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth")
    fig.suptitle(f"Well Prediction Track: {title}", y=0.995)
    save_figure(fig, out_base)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, values: pd.Series | np.ndarray, fmt: str = "{:.2f}", fontsize: int = 8) -> None:
    values = list(values)
    for patch, value in zip(ax.patches, values):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        if np.isfinite(value):
            ax.text(x, y, fmt.format(value), ha="center", va="bottom", fontsize=fontsize, rotation=90)


def resolve(path: str | Path | None, root: Path) -> Path:
    if path is None:
        return root
    p = Path(path)
    return p if p.is_absolute() else root / p


if __name__ == "__main__":
    main()
