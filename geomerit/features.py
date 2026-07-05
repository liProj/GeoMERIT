from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    label: str = "label_idx"
    well: str = "well_id"


def build_features(
    df: pd.DataFrame,
    curves: list[str],
    primary_curves: list[str],
    depth_col: str = "DEPTH_MD",
    label_col: str = "FORCE_2020_LITHOFACIES_LITHOLOGY",
    confidence_col: str = "FORCE_2020_LITHOFACIES_CONFIDENCE",
    outlier_z_threshold: float = 5.0,
    sentinel_missing: float = -999.0,
    sentinel_outlier: float = -777.0,
    window_radii: list[int] | None = None,
    add_stratigraphy: bool = True,
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Create GeoMERIT feature columns. Operations are strictly grouped by well."""
    window_radii = window_radii or [2, 5, 10]
    out = df.copy()
    for col in curves:
        if col not in out.columns:
            out[col] = np.nan

    numeric_features: list[str] = []
    categorical_features: list[str] = []

    out = out.sort_values(["well_id", depth_col], kind="mergesort").reset_index(drop=True)
    out["curve_suite_id"] = _curve_suite_id(out, curves)
    categorical_features.append("curve_suite_id")

    missing_cols = []
    outlier_cols = []
    encoded_cols = []
    for col in curves:
        miss_col = f"{col}_is_missing"
        out_col = f"{col}_is_outlier"
        enc_col = f"{col}_enc"
        z_col = f"{col}_well_robust_z"
        out[miss_col] = out[col].isna().astype(np.int8)
        out[z_col] = _robust_z_by_well(out, col)
        out[out_col] = (out[z_col].abs() > outlier_z_threshold).fillna(False).astype(np.int8)
        out[enc_col] = out[col]
        out.loc[out[miss_col] == 1, enc_col] = sentinel_missing
        out.loc[(out[miss_col] == 0) & (out[out_col] == 1), enc_col] = sentinel_outlier
        missing_cols.append(miss_col)
        outlier_cols.append(out_col)
        encoded_cols.extend([enc_col, z_col, miss_col, out_col])

    out["missing_count"] = out[missing_cols].sum(axis=1)
    out["missing_rate"] = out["missing_count"] / max(len(curves), 1)
    out["outlier_count"] = out[outlier_cols].sum(axis=1)
    out["outlier_rate"] = out["outlier_count"] / max(len(curves), 1)
    numeric_features.extend(encoded_cols + ["missing_count", "missing_rate", "outlier_count", "outlier_rate"])

    for col in primary_curves:
        if col not in out.columns:
            continue
        grad = f"{col}_grad"
        out[grad] = _gradient_by_well(out, col, depth_col)
        numeric_features.append(grad)
        for radius in window_radii:
            numeric_features.extend(_rolling_features(out, col, radius))

    numeric_features.extend(_diagnostic_features(out))
    numeric_features.extend(_spatial_features(out))

    if add_stratigraphy:
        for col in ["GROUP", "FORMATION"]:
            if col in out.columns:
                out[col] = out[col].fillna("Unknown").astype(str)
                categorical_features.append(col)
    for col in ["dist_to_nearest_casing", "is_below_deepest_casing"]:
        if col in out.columns:
            numeric_features.append(col)

    out["confidence"] = out[confidence_col] if confidence_col in out.columns else np.nan
    spec = FeatureSpec(numeric=_unique_existing(numeric_features, out), categorical=_unique_existing(categorical_features, out))
    return out, spec


def _curve_suite_id(df: pd.DataFrame, curves: list[str]) -> pd.Series:
    suite_by_well = {}
    for well, part in df.groupby("well_id", sort=False):
        bits = ["1" if col in part.columns and part[col].notna().any() else "0" for col in curves]
        suite_by_well[well] = "".join(bits)
    unique = {suite: idx for idx, suite in enumerate(sorted(set(suite_by_well.values())))}
    return df["well_id"].map(lambda w: unique[suite_by_well[w]]).astype("category")


def _robust_z_by_well(df: pd.DataFrame, col: str) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        median = values.median(skipna=True)
        mad = (values - median).abs().median(skipna=True)
        scale = mad if pd.notna(mad) and mad > 0 else values.std(skipna=True)
        if pd.isna(scale) or scale == 0:
            scale = 1.0
        return (values - median) / (scale + 1e-9)

    return df.groupby("well_id", sort=False)[col].transform(transform)


def _gradient_by_well(df: pd.DataFrame, col: str, depth_col: str) -> pd.Series:
    values = df.groupby("well_id", sort=False)[col].diff()
    depths = df.groupby("well_id", sort=False)[depth_col].diff()
    depths = depths.mask(depths.abs() < 1e-9, np.nan)
    return values / (depths + 1e-9)


def _rolling_features(df: pd.DataFrame, col: str, radius: int) -> list[str]:
    window = 2 * radius + 1
    prefix = f"{col}_w{radius}"
    names = [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max", f"{prefix}_delta"]

    grouped = df.groupby("well_id", sort=False)[col]
    rolled = grouped.rolling(window=window, min_periods=1, center=True)
    df[names[0]] = rolled.mean().reset_index(level=0, drop=True)
    df[names[1]] = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
    df[names[2]] = rolled.min().reset_index(level=0, drop=True)
    df[names[3]] = rolled.max().reset_index(level=0, drop=True)
    df[names[4]] = grouped.shift(-radius) - grouped.shift(radius)
    return names


def _diagnostic_features(df: pd.DataFrame) -> list[str]:
    made = []
    if {"RHOB", "NPHI"}.issubset(df.columns):
        df["rhob_nphi_sep"] = df["RHOB"] - df["NPHI"]
        made.append("rhob_nphi_sep")
    if {"DTC", "RHOB"}.issubset(df.columns):
        df["dtc_rhob_ratio"] = df["DTC"] / (df["RHOB"].abs() + 1e-6)
        made.append("dtc_rhob_ratio")
    if {"GR", "RDEP"}.issubset(df.columns):
        df["gr_log_rdep_ratio"] = df["GR"] / np.log1p(df["RDEP"].clip(lower=0))
        made.append("gr_log_rdep_ratio")
    if {"PEF", "RHOB"}.issubset(df.columns):
        df["pef_rhob_ratio"] = df["PEF"] / (df["RHOB"].abs() + 1e-6)
        made.append("pef_rhob_ratio")
    return made


def _spatial_features(df: pd.DataFrame) -> list[str]:
    cols = [col for col in ["x_loc", "y_loc", "z_loc"] if col in df.columns]
    made = []
    if {"x_loc", "y_loc"}.issubset(df.columns):
        cx = df["x_loc"].median(skipna=True)
        cy = df["y_loc"].median(skipna=True)
        df["dist_to_xy_centroid"] = np.sqrt((df["x_loc"] - cx) ** 2 + (df["y_loc"] - cy) ** 2)
        made.append("dist_to_xy_centroid")
    made.extend(cols)
    return made


def _unique_existing(names: list[str], df: pd.DataFrame) -> list[str]:
    seen = set()
    out = []
    for name in names:
        if name in df.columns and name not in seen:
            out.append(name)
            seen.add(name)
    return out
