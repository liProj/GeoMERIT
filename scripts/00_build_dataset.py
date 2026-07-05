from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import PerformanceWarning
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geomerit.features import build_features
from geomerit.io_las import add_casing_features, add_stratigraphy, read_las_dir, resolve_path
from geomerit.labels import encode_labels, load_penalty_matrix, save_penalty_matrix


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--features", default="configs/features.yaml")
    parser.add_argument("--las_dir", default=None)
    parser.add_argument("--out", default="data/feature_table.parquet")
    parser.add_argument("--limit_wells", type=int, default=None)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    feat_cfg = yaml.safe_load(Path(args.features).read_text(encoding="utf-8"))

    paths = data_cfg["paths"]
    las_dir = Path(args.las_dir) if args.las_dir else resolve_path(paths["las_dir"], project_root)
    df = read_las_dir(las_dir, null_value=float(data_cfg.get("null_value", -999.25)), limit=args.limit_wells)

    df = add_stratigraphy(
        df,
        resolve_path(paths["lithostrat_groups"], project_root),
        resolve_path(paths["lithostrat_formations"], project_root),
        depth_col=data_cfg["columns"]["depth"],
    )
    df = add_casing_features(
        df,
        resolve_path(paths["casing"], project_root),
        depth_col=data_cfg["columns"]["depth"],
    )

    label_col = data_cfg["columns"]["lithology"]
    df["label_idx"] = encode_labels(df[label_col]) if label_col in df.columns else -1

    features, spec = build_features(
        df,
        curves=data_cfg["curves"],
        primary_curves=data_cfg.get("primary_curves", data_cfg["curves"]),
        depth_col=data_cfg["columns"]["depth"],
        label_col=label_col,
        confidence_col=data_cfg["columns"]["confidence"],
        outlier_z_threshold=float(feat_cfg.get("outlier_z_threshold", 5.0)),
        sentinel_missing=float(feat_cfg.get("sentinel_missing", -999.0)),
        sentinel_outlier=float(feat_cfg.get("sentinel_outlier", -777.0)),
        window_radii=list(feat_cfg.get("window_radii", [2, 5, 10])),
        add_stratigraphy=bool(feat_cfg.get("add_stratigraphy_features", True)),
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = _downcast(features)
    features.to_parquet(out_path, index=False, compression=args.compression)

    meta = {
        "numeric": spec.numeric,
        "categorical": spec.categorical,
        "label": spec.label,
        "well": spec.well,
        "rows": int(len(features)),
        "labeled_rows": int((features["label_idx"] >= 0).sum()),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    penalty = load_penalty_matrix(resolve_path(paths["scoring_matrix"], project_root))
    save_penalty_matrix(project_root / "configs" / "penalty_matrix.csv", penalty)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.meta.json')}")
    print(f"Wrote {project_root / 'configs' / 'penalty_matrix.csv'}")

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in df.dtypes.items():
        if dtype == "float64":
            df[col] = df[col].astype("float32", copy=False)
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


if __name__ == "__main__":
    main()
