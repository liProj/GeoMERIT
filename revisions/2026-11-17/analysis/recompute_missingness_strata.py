"""Recompute the four natural-missingness strata and Supplementary Table S6.

Inputs are the revision dataset manifest and ten saved outer-fold predictions.
This analysis does not fit a model or construct predictions from raw LAS files.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STRATA = ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.000001))
N_CLASSES = 12
N_BOOTSTRAP = 5000
SEED = 20261103


def confusion_matrix(labels, predictions):
    return np.bincount(
        labels.astype(int) * N_CLASSES + predictions.astype(int),
        minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)


def weighted_f1(confusion):
    support = confusion.sum(axis=-1)
    predicted_support = confusion.sum(axis=-2)
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = support + predicted_support
    f1 = np.divide(
        2 * diagonal,
        denominator,
        out=np.zeros_like(diagonal, dtype=float),
        where=denominator > 0,
    )
    return (f1 * support).sum(axis=-1) / support.sum(axis=-1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Experiment root containing experiment_v1/dataset and runs")
    parser.add_argument("--report-dir", type=Path,
                        help="Optional matching report directory: verify prediction hashes against final_report.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for the three original JSON/CSV products")
    args = parser.parse_args()

    manifest = json.loads((args.root / "experiment_v1/dataset/dataset.json").read_text(encoding="utf-8"))
    names = manifest["class_names"]
    if len(names) != N_CLASSES:
        raise ValueError("This revision analysis requires the fixed 12-class ontology.")
    provenance = None
    if args.report_dir:
        report = json.loads((args.report_dir / "final_report.json").read_text(encoding="utf-8"))
        provenance = {int(row["fold"]): row["sha256"]
                      for row in report["prediction_provenance"] if row["variant"] == "full"}

    parts = []
    for fold in range(10):
        path = args.root / f"experiment_v1/runs/full/outer_{fold:02d}/predictions.parquet"
        if provenance is not None and sha256(path) != provenance[fold]:
            raise ValueError(f"Prediction SHA256 does not match the supplied report: outer_{fold:02d}")
        parts.append(pd.read_parquet(path, columns=[
            "well_id", "label_idx", "well_missing_rate", "pred_full",
        ]))
    full = pd.concat(parts, ignore_index=True)

    results = []
    for low, high in STRATA:
        subset = full[(full.well_missing_rate >= low) & (full.well_missing_rate < high)]
        groups = list(subset.groupby("well_id", sort=True))
        matrices = np.stack([
            confusion_matrix(group.label_idx.to_numpy(), group.pred_full.to_numpy())
            for _, group in groups
        ])
        n_wells = len(groups)
        # Deliberately restart the identical seed for every stratum: no offset.
        rng = np.random.default_rng(SEED)
        counts = rng.multinomial(n_wells, np.full(n_wells, 1 / n_wells), size=N_BOOTSTRAP)
        boot = weighted_f1(
            (counts @ matrices.reshape(n_wells, 144)).reshape(-1, 12, 12)
        )
        results.append(dict(
            low=low, high=high, wells=n_wells, rows=len(subset),
            weighted_f1=float(weighted_f1(matrices.sum(axis=0))),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            class_counts={names[k]: int(v) for k, v in subset.groupby("label_idx").size().items()},
            class_wells={names[k]: int(v) for k, v in subset.groupby("label_idx").well_id.nunique().items()},
            well_ids=[well for well, _ in groups],
        ))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "subgroup_current.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = [{k: v for k, v in row.items()
                if k not in ["class_counts", "class_wells", "well_ids"]} for row in results]
    pd.DataFrame(summary).to_csv(args.out / "subgroup_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"missingness_low": row["low"], "missingness_high": row["high"], "class": name,
         "samples": row["class_counts"].get(name, 0), "wells": row["class_wells"].get(name, 0)}
        for row in results for name in names
    ]).to_csv(args.out / "subgroup_class_support.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
