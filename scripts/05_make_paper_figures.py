from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomerit.decode import normalize_proba
from geomerit.labels import LABEL_NAMES, TAIL_IDS, load_penalty_matrix
from geomerit.metrics import boundary_mask_by_well


METHOD_COLORS = {
    "Paper Baseline XGBoost": "#8f9aa8",
    "Paper RFE + XGBoost": "#4c78a8",
    "GeoMERIT 3GBDT base": "#72b7b2",
    "GeoRACS best": "#f58518",
    "GeoRACS F1-oriented": "#e45756",
    "Retrieval prior only": "#b279a2",
    "Stacking with retrieval": "#54a24b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style figures and source data for GeoRACS.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--penalty", default="configs/penalty_matrix.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    table_path = resolve_path(args.table, project_root)
    run_dir = resolve_path(args.run, project_root)
    out_dir = resolve_path(args.out, project_root)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    A = load_penalty_matrix(resolve_path(args.penalty, project_root)).astype(np.float32)
    rows = pd.read_parquet(run_dir / "oof_rows.parquet")
    y_true = rows["label_idx"].to_numpy(np.int16)
    well_id = rows["well_id"].astype(str).to_numpy()
    depth = rows["DEPTH_MD"].to_numpy(np.float32)
    boundary_mask = boundary_mask_by_well(y_true, well_id, radius=1)
    proba_base = normalize_proba(np.load(run_dir / "oof_proba.npy")).astype(np.float32)
    pred_base = bayes_decode(proba_base, A)
    pred_georacs = np.load(run_dir / "georacs_oof_report.pred.npy").astype(np.int16)
    pred_retrieval = bayes_decode(normalize_proba(np.load(run_dir / "georacs_retrieval_prior_k64_ivf.npy")), A)
    pred_stack = bayes_decode(normalize_proba(np.load(run_dir / "georacs_stack_oof.npy")), A)

    report = json.loads((run_dir / "georacs_oof_report.json").read_text(encoding="utf-8"))
    dol_report = load_json_if_exists(run_dir / "georacs_dolomite_carbonate_expert_report.json")
    physics = load_json_if_exists(run_dir / "georacs_physics_feature_summary.json")

    metrics_by_method = build_main_metrics(
        y_true,
        well_id,
        A,
        boundary_mask,
        pred_base,
        pred_georacs,
        pred_retrieval,
        pred_stack,
        report,
    )
    save_table(metrics_by_method, data_dir / "main_metrics_comparison.csv")
    plot_main_metrics(metrics_by_method, fig_dir / "fig03_main_metrics_comparison")

    class_distribution = build_class_distribution(y_true)
    save_table(class_distribution, data_dir / "class_distribution.csv")
    plot_class_distribution(class_distribution, fig_dir / "fig01_class_distribution")

    per_class = build_per_class(y_true, pred_base, pred_georacs)
    save_table(per_class, data_dir / "per_class_base_vs_georacs.csv")
    plot_per_class(per_class, fig_dir / "fig05_per_class_f1_recall")
    plot_tail_classes(per_class, fig_dir / "fig06_tail_class_f1_comparison")

    ablation = build_ablation_table(metrics_by_method, report, dol_report)
    save_table(ablation, data_dir / "ablation_and_module_candidates.csv")
    plot_ablation(ablation, fig_dir / "fig04_ablation_module_contribution")
    plot_tradeoff(ablation, fig_dir / "fig08_penalty_macro_tradeoff")

    confusion_base = normalized_confusion(y_true, pred_base)
    confusion_georacs = normalized_confusion(y_true, pred_georacs)
    save_table(confusion_base, data_dir / "confusion_base_bayes_normalized.csv")
    save_table(confusion_georacs, data_dir / "confusion_georacs_best_normalized.csv")
    plot_confusion(confusion_georacs, fig_dir / "fig10_confusion_matrix_georacs")

    missing_outputs = build_missing_outputs(table_path, run_dir, y_true, pred_base, pred_georacs, A, boundary_mask)
    for name, table in missing_outputs.items():
        save_table(table, data_dir / f"{name}.csv")
    plot_missing_heatmap(missing_outputs["missing_by_well_curve"], fig_dir / "fig02_missing_heatmap_by_well_curve")
    plot_robustness(missing_outputs["robustness_by_missing_threshold"], fig_dir / "fig07_missing_robustness")

    if physics:
        physics_table = flatten_physics_summary(physics)
        save_table(physics_table, data_dir / "physics_feature_summary.csv")
        plot_physics_overlap(physics_table, fig_dir / "fig09_carbonate_physics_overlap")

    prediction_rows = pd.DataFrame(
        {
            "well_id": well_id,
            "depth": depth,
            "y_true": y_true,
            "pred_base_bayes": pred_base,
            "pred_georacs_best": pred_georacs,
            "true_name": [LABEL_NAMES[int(v)] for v in y_true],
            "base_name": [LABEL_NAMES[int(v)] for v in pred_base],
            "georacs_name": [LABEL_NAMES[int(v)] for v in pred_georacs],
        }
    )
    prediction_rows.to_csv(data_dir / "oof_predictions_base_georacs.csv", index=False)

    manifest = {
        "figures": sorted(str(path.relative_to(out_dir)) for path in fig_dir.glob("*")),
        "data": sorted(str(path.relative_to(out_dir)) for path in data_dir.glob("*")),
        "notes": [
            "Paper values are taken from the user-provided RFE paper summary.",
            "GeoMERIT 3GBDT base is Bayes-risk decoding of the completed 10-fold OOF probabilities.",
            "GeoRACS best is temperature calibration + logit adjustment + tail expert gate + Bayes risk.",
            "Retrieval and stacking are included as diagnostics because this run showed they hurt the objective.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.make_archive(str(out_dir), "zip", out_dir)
    print(json.dumps({"out_dir": str(out_dir), "zip": str(out_dir) + ".zip", **manifest}, indent=2), flush=True)


def build_main_metrics(
    y_true: np.ndarray,
    well_id: np.ndarray,
    A: np.ndarray,
    boundary_mask: np.ndarray,
    pred_base: np.ndarray,
    pred_georacs: np.ndarray,
    pred_retrieval: np.ndarray,
    pred_stack: np.ndarray,
    report: dict,
) -> pd.DataFrame:
    rows = [
        {
            "method": "Paper Baseline XGBoost",
            "weighted_f1": 0.664,
            "macro_f1": np.nan,
            "boundary_f1": 0.351,
            "penalty": -0.780,
            "tail_mean_f1": np.nan,
            "source": "RFE paper summary",
        },
        {
            "method": "Paper RFE + XGBoost",
            "weighted_f1": 0.7272,
            "macro_f1": np.nan,
            "boundary_f1": 0.410,
            "penalty": -0.6289,
            "tail_mean_f1": np.nan,
            "source": "RFE paper summary",
        },
    ]
    rows.append({"method": "GeoMERIT 3GBDT base", **evaluate(y_true, pred_base, A, boundary_mask), "source": "current OOF"})
    rows.append({"method": "GeoRACS best", **evaluate(y_true, pred_georacs, A, boundary_mask), "source": "current OOF"})
    rows.append({"method": "Retrieval prior only", **evaluate(y_true, pred_retrieval, A, boundary_mask), "source": "diagnostic"})
    rows.append({"method": "Stacking with retrieval", **evaluate(y_true, pred_stack, A, boundary_mask), "source": "diagnostic"})
    f1_oriented = find_candidate(report, "blend_eta=0_tau=0.2_gamma=1_theta=0.35_bayes")
    if f1_oriented:
        rows.append(
            {
                "method": "GeoRACS F1-oriented",
                "weighted_f1": f1_oriented["weighted_f1"],
                "macro_f1": f1_oriented["macro_f1"],
                "boundary_f1": f1_oriented["boundary_f1"],
                "penalty": f1_oriented["penalty"],
                "tail_mean_f1": f1_oriented["tail_mean_f1"],
                "source": "current OOF candidate",
            }
        )
    return pd.DataFrame(rows)


def build_class_distribution(y_true: np.ndarray) -> pd.DataFrame:
    counts = np.bincount(y_true, minlength=len(LABEL_NAMES))
    total = counts.sum()
    return pd.DataFrame(
        {
            "class_id": np.arange(len(LABEL_NAMES)),
            "class_name": LABEL_NAMES,
            "support": counts,
            "fraction": counts / total,
        }
    )


def build_per_class(y_true: np.ndarray, pred_base: np.ndarray, pred_georacs: np.ndarray) -> pd.DataFrame:
    frames = []
    for method, pred in [("GeoMERIT 3GBDT base", pred_base), ("GeoRACS best", pred_georacs)]:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, pred, labels=list(range(len(LABEL_NAMES))), zero_division=0
        )
        frame = pd.DataFrame(
            {
                "method": method,
                "class_id": np.arange(len(LABEL_NAMES)),
                "class_name": LABEL_NAMES,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "is_tail": [i in TAIL_IDS for i in range(len(LABEL_NAMES))],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_ablation_table(metrics_by_method: pd.DataFrame, report: dict, dol_report: list[dict] | None) -> pd.DataFrame:
    rows = []
    rows.extend(
        [
            {
                "component": "Paper Baseline XGBoost",
                "weighted_f1": 0.664,
                "macro_f1": np.nan,
                "boundary_f1": 0.351,
                "penalty": -0.780,
                "tail_mean_f1": np.nan,
                "notes": "paper baseline",
            },
            {
                "component": "Paper + Weighting",
                "weighted_f1": 0.7230,
                "macro_f1": np.nan,
                "boundary_f1": np.nan,
                "penalty": -0.6346,
                "tail_mean_f1": np.nan,
                "notes": "paper ablation",
            },
            {
                "component": "Paper + RFE FE",
                "weighted_f1": 0.7259,
                "macro_f1": np.nan,
                "boundary_f1": np.nan,
                "penalty": -0.6327,
                "tail_mean_f1": np.nan,
                "notes": "paper ablation",
            },
            {
                "component": "Paper RFE + XGBoost",
                "weighted_f1": 0.7272,
                "macro_f1": np.nan,
                "boundary_f1": 0.410,
                "penalty": -0.6289,
                "tail_mean_f1": np.nan,
                "notes": "paper full",
            },
        ]
    )
    candidate_names = [
        ("GeoMERIT 3GBDT base", "base_bayes"),
        ("+ temperature calibration", "base_temp_bayes"),
        ("+ logit adjust + tail gate", report["best"]["name"]),
    ]
    for label, name in candidate_names:
        candidate = find_candidate(report, name)
        if candidate is None and name == report["best"]["name"]:
            candidate = report["best"]
        if candidate is None:
            continue
        rows.append(
            {
                "component": label,
                "weighted_f1": candidate["weighted_f1"],
                "macro_f1": candidate["macro_f1"],
                "boundary_f1": candidate["boundary_f1"],
                "penalty": candidate["penalty"],
                "tail_mean_f1": candidate["tail_mean_f1"],
                "notes": name,
            }
        )
    for diagnostic in ["Retrieval prior only", "Stacking with retrieval", "GeoRACS F1-oriented"]:
        match = metrics_by_method[metrics_by_method["method"] == diagnostic]
        if match.empty:
            continue
        item = match.iloc[0]
        rows.append(
            {
                "component": diagnostic,
                "weighted_f1": item["weighted_f1"],
                "macro_f1": item["macro_f1"],
                "boundary_f1": item["boundary_f1"],
                "penalty": item["penalty"],
                "tail_mean_f1": item["tail_mean_f1"],
                "notes": item["source"],
            }
        )
    if dol_report:
        best_dol = max(dol_report, key=lambda item: item.get("dolomite", 0.0))
        rows.append(
            {
                "component": "Dolomite carbonate expert max-Dolomite",
                "weighted_f1": best_dol["weighted"],
                "macro_f1": best_dol["macro"],
                "boundary_f1": best_dol["boundary"],
                "penalty": best_dol["penalty"],
                "tail_mean_f1": best_dol["tail"],
                "notes": f"tau={best_dol['tau']}, g={best_dol['g']}, th={best_dol['th']}, dolomite_f1={best_dol['dolomite']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def build_missing_outputs(
    table_path: Path,
    run_dir: Path,
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_georacs: np.ndarray,
    A: np.ndarray,
    boundary_mask: np.ndarray,
) -> dict[str, pd.DataFrame]:
    meta = json.loads(table_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    missing_cols = [col for col in meta["numeric"] if col.endswith("_is_missing")]
    oof_index = np.load(run_dir / "oof_index.npy")
    df = pd.read_parquet(table_path, columns=["well_id"] + missing_cols).iloc[oof_index]
    missing_values = df[missing_cols].astype(np.float32)
    row_missing_rate = missing_values.mean(axis=1).to_numpy(np.float32)
    well_missing = (
        pd.DataFrame({"well_id": df["well_id"].astype(str), "row_missing_rate": row_missing_rate})
        .groupby("well_id", as_index=False)["row_missing_rate"]
        .mean()
        .rename(columns={"row_missing_rate": "well_missing_rate"})
    )
    heat = df.assign(well_id=df["well_id"].astype(str)).groupby("well_id")[missing_cols].mean()
    heat.columns = [col.replace("_is_missing", "") for col in heat.columns]
    heat = heat.reset_index()
    curve = heat.drop(columns=["well_id"]).mean(axis=0).rename("missing_rate").reset_index()
    curve.columns = ["curve", "missing_rate"]

    row_df = pd.DataFrame({"well_id": df["well_id"].astype(str), "row_missing_rate": row_missing_rate})
    row_df = row_df.merge(well_missing, on="well_id", how="left")
    robustness_rows = []
    for complete_th, deficient_th in [(0.1, 0.4), (0.2, 0.5), (0.3, 0.6)]:
        masks = {
            "feature_complete": row_df["well_missing_rate"].to_numpy() <= complete_th,
            "feature_deficient": row_df["well_missing_rate"].to_numpy() >= deficient_th,
        }
        for group_name, mask in masks.items():
            for method, pred in [("GeoMERIT 3GBDT base", pred_base), ("GeoRACS best", pred_georacs)]:
                if not np.any(mask):
                    continue
                local_boundary = boundary_mask[mask]
                local_metrics = evaluate(y_true[mask], pred[mask], A, local_boundary)
                robustness_rows.append(
                    {
                        "complete_threshold": complete_th,
                        "deficient_threshold": deficient_th,
                        "group": group_name,
                        "method": method,
                        "rows": int(mask.sum()),
                        **local_metrics,
                    }
                )
    return {
        "missing_by_well_curve": heat,
        "curve_missing_rate": curve.sort_values("missing_rate", ascending=False),
        "well_missing_rate": well_missing.sort_values("well_missing_rate", ascending=False),
        "row_missing_rate": row_df,
        "robustness_by_missing_threshold": pd.DataFrame(robustness_rows),
    }


def flatten_physics_summary(summary: dict) -> pd.DataFrame:
    rows = []
    for class_id, item in summary.items():
        for feature, stats in item.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": item["name"],
                    "feature": feature,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def plot_class_distribution(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(df["class_name"], df["support"], color="#4c78a8")
    ax.set_yscale("log")
    ax.set_ylabel("Support (log scale)")
    ax.set_title("FORCE 2020 OOF Label Distribution")
    ax.tick_params(axis="x", rotation=45)
    for bar, value in zip(bars, df["support"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{int(value):,}", ha="center", va="bottom", fontsize=7)
    save_figure(fig, stem)


def plot_main_metrics(df: pd.DataFrame, stem: Path) -> None:
    metrics = ["weighted_f1", "macro_f1", "boundary_f1", "penalty", "tail_mean_f1"]
    labels = ["Weighted F1", "Macro F1", "Boundary F1", "Penalty", "Tail mean F1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.8), constrained_layout=True)
    for ax, metric, label in zip(axes, metrics, labels):
        sub = df.dropna(subset=[metric])
        colors = [METHOD_COLORS.get(name, "#999999") for name in sub["method"]]
        ax.barh(sub["method"], sub[metric], color=colors)
        ax.set_title(label)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="x", alpha=0.25)
        for y, value in enumerate(sub[metric]):
            ha = "left" if value >= 0 else "right"
            dx = 0.005 if value >= 0 else -0.005
            ax.text(value + dx, y, f"{value:.3f}", va="center", ha=ha, fontsize=8)
    fig.suptitle("Main Metric Comparison: Paper Baselines vs Current GeoRACS")
    save_figure(fig, stem)


def plot_per_class(df: pd.DataFrame, stem: Path) -> None:
    classes = LABEL_NAMES
    x = np.arange(len(classes))
    width = 0.36
    base = df[df["method"] == "GeoMERIT 3GBDT base"].set_index("class_name").loc[classes]
    geo = df[df["method"] == "GeoRACS best"].set_index("class_name").loc[classes]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].bar(x - width / 2, base["f1"], width, label="GeoMERIT base", color="#72b7b2")
    axes[0].bar(x + width / 2, geo["f1"], width, label="GeoRACS best", color="#f58518")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x - width / 2, base["recall"], width, label="GeoMERIT base", color="#72b7b2")
    axes[1].bar(x + width / 2, geo["recall"], width, label="GeoRACS best", color="#f58518")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=35, ha="right")
    fig.suptitle("Per-Class F1 and Recall")
    save_figure(fig, stem)


def plot_tail_classes(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["class_id"].isin(TAIL_IDS)]
    classes = [LABEL_NAMES[i] for i in TAIL_IDS]
    base = sub[sub["method"] == "GeoMERIT 3GBDT base"].set_index("class_name").loc[classes]
    geo = sub[sub["method"] == "GeoRACS best"].set_index("class_name").loc[classes]
    x = np.arange(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, base["f1"], width, label="GeoMERIT base", color="#72b7b2")
    ax.bar(x + width / 2, geo["f1"], width, label="GeoRACS best", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1")
    ax.set_title("Tail-Class F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, stem)


def plot_ablation(df: pd.DataFrame, stem: Path) -> None:
    sub = df.copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(sub))
    ax.barh(y - 0.18, sub["weighted_f1"], height=0.18, label="Weighted F1", color="#4c78a8")
    ax.barh(y, sub["macro_f1"], height=0.18, label="Macro F1", color="#f58518")
    ax.barh(y + 0.18, sub["boundary_f1"], height=0.18, label="Boundary F1", color="#54a24b")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["component"])
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Score")
    ax.set_title("Ablation and Module Contribution")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(fig, stem)


def plot_tradeoff(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sub = df.dropna(subset=["macro_f1", "penalty"])
    colors = ["#f58518" if "GeoRACS" in name else "#4c78a8" if "Paper" in name else "#8f9aa8" for name in sub["component"]]
    ax.scatter(sub["penalty"], sub["macro_f1"], s=80, c=colors, edgecolor="white", linewidth=0.8)
    for _, row in sub.iterrows():
        ax.annotate(row["component"], (row["penalty"], row["macro_f1"]), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Penalty score (closer to 0 is better)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Penalty vs Macro-F1 Trade-Off")
    ax.grid(alpha=0.25)
    save_figure(fig, stem)


def plot_missing_heatmap(df: pd.DataFrame, stem: Path) -> None:
    heat = df.set_index("well_id")
    order = heat.mean(axis=1).sort_values().index
    heat = heat.loc[order]
    max_wells = 80
    if len(heat) > max_wells:
        positions = np.linspace(0, len(heat) - 1, max_wells).astype(int)
        heat = heat.iloc[positions]
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(heat.to_numpy(float), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index, fontsize=6)
    ax.set_title("Missing-Rate Heatmap by Well and Curve")
    ax.set_xlabel("Curve")
    ax.set_ylabel("Well")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Missing rate")
    save_figure(fig, stem)


def plot_robustness(df: pd.DataFrame, stem: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in [(axes[0], "weighted_f1", "Weighted F1"), (axes[1], "boundary_f1", "Boundary F1")]:
        labels = []
        x = np.arange(len(df[["complete_threshold", "deficient_threshold", "group"]].drop_duplicates()))
        for i, ((ct, dt, group), part) in enumerate(df.groupby(["complete_threshold", "deficient_threshold", "group"], sort=False)):
            labels.append(f"{group}\n{ct}/{dt}")
            for j, method in enumerate(["GeoMERIT 3GBDT base", "GeoRACS best"]):
                val = part.loc[part["method"] == method, metric]
                if not val.empty:
                    ax.bar(i + (j - 0.5) * 0.32, float(val.iloc[0]), width=0.30, color="#72b7b2" if j == 0 else "#f58518", label=method if i == 0 else None)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Robustness on Feature-Complete vs Feature-Deficient Wells")
    save_figure(fig, stem)


def plot_confusion(df: pd.DataFrame, stem: Path) -> None:
    mat = df.set_index("true_class_name")[LABEL_NAMES].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(LABEL_NAMES)))
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("GeoRACS Best Confusion Matrix (Row-Normalized)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= 0.05:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] > 0.45 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_figure(fig, stem)


def plot_physics_overlap(df: pd.DataFrame, stem: Path) -> None:
    classes = ["Limestone", "Marl", "Chalk", "Dolomite", "Anhydrite", "Basement"]
    features = ["PEF_enc", "RHOB_enc", "NPHI_enc", "DTC_enc", "GR_enc"]
    fig, axes = plt.subplots(len(features), 1, figsize=(11, 11), constrained_layout=True)
    for ax, feature in zip(axes, features):
        sub = df[(df["feature"] == feature) & (df["class_name"].isin(classes))].set_index("class_name").reindex(classes)
        x = np.arange(len(classes))
        med = sub["p50"].to_numpy(float)
        lo = sub["p10"].to_numpy(float)
        hi = sub["p90"].to_numpy(float)
        lower = np.where(np.isfinite(med - lo), med - lo, 0)
        upper = np.where(np.isfinite(hi - med), hi - med, 0)
        ax.errorbar(x, med, yerr=[lower, upper], fmt="o", color="#4c78a8", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Carbonate/Evaporite Physics Feature Overlap (p10/p50/p90)")
    save_figure(fig, stem)


def normalized_confusion(y_true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, pred, labels=list(range(len(LABEL_NAMES)))).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    df = pd.DataFrame(cm, columns=LABEL_NAMES)
    df.insert(0, "true_class_name", LABEL_NAMES)
    df.insert(0, "true_class_id", np.arange(len(LABEL_NAMES)))
    return df


def evaluate(y_true: np.ndarray, pred: np.ndarray, A: np.ndarray, boundary_mask: np.ndarray) -> dict[str, float]:
    tails = f1_score(y_true, pred, labels=TAIL_IDS, average=None, zero_division=0)
    boundary = f1_score(y_true[boundary_mask], pred[boundary_mask], average="weighted", zero_division=0) if np.any(boundary_mask) else np.nan
    return {
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(len(LABEL_NAMES))), average="macro", zero_division=0)),
        "boundary_f1": float(boundary),
        "penalty": float(-np.mean(A[y_true.astype(int), pred.astype(int)])),
        "tail_mean_f1": float(np.mean(tails)),
    }


def bayes_decode(proba: np.ndarray, A: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.zeros(len(proba), dtype=np.int16)
    for start in range(0, len(proba), chunk):
        end = min(start + chunk, len(proba))
        out[start:end] = (normalize_proba(proba[start:end]) @ A).argmin(axis=1).astype(np.int16)
    return out


def find_candidate(report: dict, name: str) -> dict | None:
    if report.get("best", {}).get("name") == name:
        return report["best"]
    for candidate in report.get("candidates", []):
        if candidate.get("name") == name:
            return candidate
    return None


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


if __name__ == "__main__":
    main()
