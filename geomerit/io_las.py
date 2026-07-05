from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_las_header(path: str | Path) -> tuple[list[str], dict[str, str], int]:
    """Parse curve names, selected well metadata, and the first ASCII data line."""
    path = Path(path)
    curves: list[str] = []
    meta: dict[str, str] = {}
    in_curve = False
    ascii_line = -1
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("~curve"):
                in_curve = True
                continue
            if lower.startswith("~ascii"):
                ascii_line = line_no
                break
            if stripped.startswith("~"):
                in_curve = False
            if in_curve and stripped and not stripped.startswith("#"):
                left = stripped.split(":", 1)[0].strip()
                name = left.split(".", 1)[0].strip()
                if name:
                    curves.append(name)
                continue
            if stripped and not stripped.startswith("#") and "." in stripped and ":" in stripped:
                key = stripped.split(".", 1)[0].strip()
                value = stripped.split(".", 1)[1].split(":", 1)[0].strip()
                if key:
                    meta[key] = value
    if ascii_line < 0:
        raise ValueError(f"LAS file has no ~Ascii section: {path}")
    return curves, meta, ascii_line


def read_las_file(path: str | Path, null_value: float = -999.25) -> pd.DataFrame:
    """Read a LAS file into a DataFrame without requiring lasio."""
    path = Path(path)
    curves, meta, ascii_line = parse_las_header(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=curves,
        skiprows=ascii_line + 1,
        comment="#",
        engine="c",
        na_values=[null_value],
    )
    well_id = meta.get("UWI") or meta.get("WELL") or path.stem.replace("_", "/")
    df.insert(0, "well_id", well_id)
    df.insert(1, "source_file", path.name)
    if "DEPTH_MD" not in df.columns and "DEPT" in df.columns:
        df["DEPTH_MD"] = df["DEPT"]
    return df


def read_las_dir(las_dir: str | Path, null_value: float = -999.25, limit: int | None = None) -> pd.DataFrame:
    las_dir = Path(las_dir)
    files = sorted(las_dir.glob("*.las"))
    if limit is not None:
        files = files[:limit]
    frames = []
    for path in files:
        frames.append(read_las_file(path, null_value=null_value))
    if not frames:
        raise FileNotFoundError(f"No .las files found in {las_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_excel_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_excel(path)


def add_stratigraphy(
    data: pd.DataFrame,
    groups_path: str | Path | None,
    formations_path: str | Path | None,
    well_col: str = "well_id",
    depth_col: str = "DEPTH_MD",
) -> pd.DataFrame:
    out = data.copy()
    groups = load_excel_table(groups_path)
    formations = load_excel_table(formations_path)
    if groups is not None:
        out["GROUP"] = _assign_interval_surface(out, groups, well_col, depth_col, "GROUP")
    else:
        out["GROUP"] = "Unknown"
    if formations is not None:
        out["FORMATION"] = _assign_interval_surface(out, formations, well_col, depth_col, "FORMATION")
    else:
        out["FORMATION"] = "Unknown"
    return out


def add_casing_features(
    data: pd.DataFrame,
    casing_path: str | Path | None,
    well_col: str = "well_id",
    depth_col: str = "DEPTH_MD",
) -> pd.DataFrame:
    out = data.copy()
    casing = load_excel_table(casing_path)
    if casing is None or "Well identifier" not in casing.columns:
        out["dist_to_nearest_casing"] = np.nan
        out["is_below_deepest_casing"] = 0
        return out

    casing = casing.dropna(subset=["Well identifier", "MD"]).copy()
    casing["Well identifier"] = casing["Well identifier"].astype(str).str.strip()
    out["dist_to_nearest_casing"] = np.nan
    out["is_below_deepest_casing"] = 0
    for well, idx in out.groupby(well_col).groups.items():
        depths = out.loc[idx, depth_col].to_numpy(float)
        casing_depths = casing.loc[casing["Well identifier"] == str(well).strip(), "MD"].to_numpy(float)
        if len(casing_depths) == 0:
            continue
        nearest = np.min(np.abs(depths[:, None] - casing_depths[None, :]), axis=1)
        out.loc[idx, "dist_to_nearest_casing"] = nearest
        out.loc[idx, "is_below_deepest_casing"] = (depths >= np.nanmax(casing_depths)).astype(np.int8)
    return out


def _assign_interval_surface(
    data: pd.DataFrame,
    tops: pd.DataFrame,
    well_col: str,
    depth_col: str,
    output_name: str,
) -> pd.Series:
    result = pd.Series("Unknown", index=data.index, dtype="object")
    required = {"Well identifier", "MD", "Surface"}
    if not required.issubset(tops.columns):
        return result
    tops = tops.dropna(subset=["Well identifier", "MD", "Surface"]).copy()
    tops["Well identifier"] = tops["Well identifier"].astype(str).str.strip()
    tops["Surface"] = tops["Surface"].astype(str).str.replace(r"\s+Top$", "", regex=True).str.strip()
    for well, idx in data.groupby(well_col).groups.items():
        well_tops = tops.loc[tops["Well identifier"] == str(well).strip()].sort_values("MD")
        if well_tops.empty:
            continue
        top_depths = well_tops["MD"].to_numpy(float)
        names = well_tops["Surface"].to_numpy(str)
        depths = data.loc[idx, depth_col].to_numpy(float)
        pos = np.searchsorted(top_depths, depths, side="right") - 1
        valid = pos >= 0
        assigned = np.full(len(idx), "Unknown", dtype=object)
        assigned[valid] = names[pos[valid]]
        result.loc[idx] = assigned
    result.name = output_name
    return result


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def existing_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [col for col in cols if col in df.columns]
