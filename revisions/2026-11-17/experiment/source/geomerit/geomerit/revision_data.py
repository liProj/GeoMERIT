"""Revision dataset and training-only feature state.

Whole-well statistics are deliberately transductive, label-free, offline inputs.
Category vocabularies and the fleet coordinate center are fitted separately for
every inner/outer training partition. Labels never enter feature construction.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .features import build_features
from .io_las import read_las_file, add_stratigraphy, add_casing_features, resolve_path
from .labels import encode_labels, LABEL_CODES, LABEL_NAMES

COORDS = ['x_loc', 'y_loc', 'z_loc']
CASING = ['dist_to_nearest_casing', 'is_below_deepest_casing']
STRATA = ['GROUP', 'FORMATION']
META_NUMERIC = COORDS + ['dist_to_xy_centroid'] + CASING


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rfe_features(frame, curves, radius=2):
    """Paper-equation reimplementation; unresolved choices are recorded in config.

    RFE uses flattened encoded windows, stabilized raw gradients, and aggregate
    missing/outlier/padding features, not GeoMERIT's robust-z/ratio feature stack.
    """
    raw = frame[curves].astype(float)
    scale = raw.std(ddof=0).replace(0, 1)
    outliers = ((raw - raw.mean()) / scale).abs() > 5
    missing = raw.isna()
    encoded = raw.mask(outliers, -777.0).fillna(-999.0)
    depth = frame['DEPTH_MD']
    segments = ((depth.diff().isna()) | (depth.diff() <= 0) | (depth.diff() > 1.0)).cumsum()
    feats = {}
    padded = np.zeros(len(frame), dtype=float)
    for shift in range(-radius, radius + 1):
        shifted = encoded.groupby(segments, sort=False).shift(shift)
        pad = shifted.isna().all(axis=1).to_numpy()
        padded += pad
        shifted = shifted.fillna(-999.0)
        for col in curves:
            feats[f'rfe_{col}_lag{shift}'] = shifted[col].to_numpy()
    dd = depth.groupby(segments, sort=False).diff()
    for col in curves:
        feats[f'rfe_{col}_grad'] = (raw[col].groupby(segments, sort=False).diff() / (dd + 1e-6)).to_numpy()
    feats.update(rfe_missing_count=missing.sum(axis=1).to_numpy(),
                 rfe_outlier_count=outliers.sum(axis=1).to_numpy(),
                 rfe_missing_rate=missing.mean(axis=1).to_numpy(),
                 rfe_outlier_rate=outliers.mean(axis=1).to_numpy(),
                 rfe_padding_any=(padded > 0).astype(float),
                 rfe_padding_rate=padded / (2 * radius + 1))
    return pd.DataFrame(feats, index=frame.index, dtype='float32')


def load_cohort_reference(project_root):
    """Load the recorded cohort checks without requiring legacy predictions.

    The compact publication record contains the exact original well folds and
    per-well class counts. It affects cohort validation only, not features.
    """
    project_root = Path(project_root)
    snapshot = project_root / 'runs/full10_fullfeat_3gbdt_c2f_tail_complete_remote_snapshot'
    legacy = [snapshot / 'oof_rows.parquet', snapshot / 'oof_fold.npy']
    if any(p.exists() for p in legacy):
        if not all(p.is_file() for p in legacy):
            raise FileNotFoundError('Both legacy cohort reference files are required together.')
        old = pd.read_parquet(legacy[0], columns=['well_id', 'label_idx'])
        old['outer_fold'] = np.load(legacy[1])
        assert old.groupby('well_id')['outer_fold'].nunique().eq(1).all()
        return (old.groupby('well_id')['outer_fold'].first().to_dict(),
                old.groupby(['well_id', 'label_idx']).size().to_dict(), len(old), legacy)
    path = project_root / 'configs/cohort_reference.json'
    reference = json.loads(path.read_text(encoding='utf-8'))
    assert reference['class_codes'] == LABEL_CODES
    assert reference['class_names'] == LABEL_NAMES
    well_folds, counts = {}, {}
    for record in reference['well_records']:
        well, fold, values = record['well_id'], record['outer_fold'], record['class_counts']
        assert well not in well_folds and isinstance(fold, int) and 0 <= fold < 10
        assert len(values) == 12 and all(isinstance(n, int) and n >= 0 for n in values)
        well_folds[well] = fold
        counts.update({(well, c): n for c, n in enumerate(values) if n})
    expected = sum(counts.values())
    assert len(well_folds) == reference['wells'] == 118
    assert expected == reference['rows'] == 1431383
    return well_folds, counts, expected, [path]


def build_dataset(project_root, cache_dir, config):
    project_root, cache_dir = Path(project_root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'dataset.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') == 'complete':
            return manifest
        raise RuntimeError('Incomplete dataset cache; inspect it before rebuilding.')
    cfg = yaml.safe_load((project_root / 'configs/data.yaml').read_text(encoding='utf-8'))
    curves = [c for c in cfg['curves'] if c not in COORDS]
    assert len(curves) == 20
    paths = {k: resolve_path(v, project_root) for k, v in cfg['paths'].items()}
    well_folds, old_counts, expected, reference_inputs = load_cohort_reference(project_root)
    (cache_dir / 'outer_wells.json').write_text(json.dumps(well_folds, indent=2), encoding='utf-8')
    target = cache_dir / 'features.parquet'
    temporary = cache_dir / 'features.parquet.partial'
    inputs = []
    counts, total, writer, schema, geo_numeric, categories = {}, 0, None, None, None, None
    depth_filled = 0
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    manifest_path.write_text(json.dumps({'status': 'building'}))
    try:
        for number, path in enumerate(sorted(paths['las_dir'].glob('*.las')), 1):
            df = read_las_file(path)
            well = str(df['well_id'].iloc[0])
            if well not in well_folds:
                continue
            if not np.isfinite(df['DEPTH_MD']).all():
                raise ValueError(f'Nonfinite depth remains after DEPT fallback: {well}')
            for col in curves + COORDS:
                if col not in df:
                    df[col] = np.nan
            df = add_stratigraphy(df, paths['lithostrat_groups'], paths['lithostrat_formations'])
            df = add_casing_features(df, paths['casing'])
            df, spec = build_features(df, curves, cfg['primary_curves'], context_gap_max=1.0)
            observed_columns = []
            for col in curves:
                name = col + '_observed_enc'
                df[name] = df[col].mask(df[col + '_is_outlier'].eq(1), -777.0)
                observed_columns.append(name)
            df['label_idx'] = encode_labels(df[cfg['columns']['lithology']])
            engineered = rfe_features(df, curves, int(config['features']['rfe_window_radius']))
            df = pd.concat([df, engineered], axis=1)
            geo_numeric, categories = spec.numeric, spec.categorical
            columns = list(dict.fromkeys(['well_id', 'source_file', 'source_row', 'DEPTH_MD', 'label_idx', 'confidence'] + curves + observed_columns + spec.numeric + spec.categorical + list(engineered.columns)))
            df = df.loc[df['label_idx'] >= 0, columns].copy()
            df['outer_fold'] = int(well_folds[well])
            df['sample_id'] = df['source_file'] + ':' + df['source_row'].astype(str)
            df['well_missing_rate'] = float(df[curves].isna().to_numpy().mean())
            for label, n in df.groupby('label_idx').size().items():
                counts[(well, int(label))] = int(n)
            for col in df:
                if col in ['well_id', 'source_file', 'sample_id'] + categories:
                    df[col] = df[col].astype('string').fillna('Unknown')
                elif col in ['source_row', 'label_idx', 'outer_fold']:
                    df[col] = df[col].astype('int32')
                elif col == 'DEPTH_MD':
                    df[col] = df[col].astype('float64')
                else:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype('float32')
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(temporary, schema, compression='zstd')
            writer.write_table(table.cast(schema))
            total += len(df)
            inputs.append({'path': str(path), 'sha256': digest(path)})
            print(f'DATASET {number}/118 well={well} rows={len(df)} total={total}', flush=True)
        assert total == expected == 1431383, (total, expected)
        assert counts == old_counts, 'Per-well class counts do not match the historical cohort.'
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(target)
    for p in [*(p for p in paths.values() if p.is_file()), *reference_inputs]:
        if p.is_file(): inputs.append({'path': str(p), 'sha256': digest(p)})
    manifest = {'status': 'complete', 'rows': total, 'wells': len(well_folds), 'curves': curves,
                'geo_numeric': geo_numeric, 'categorical': categories,
                'rfe_numeric': list(engineered.columns), 'inputs': inputs,
                'class_codes': LABEL_CODES, 'class_names': LABEL_NAMES,
                'parquet_sha256': digest(target), 'context_gap_max_m': 1.0,
                'feature_semantics': 'offline full-well, label-free; training-only categorical vocabulary and fleet center',
                'rfe_status': 'equation-based reimplementation; radius=2 and within-well standard-z are explicit assumptions, not exact original code'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def feature_columns(manifest, variant):
    family = variant.get('family', 'geo')
    numeric = list(manifest['geo_numeric'] if family == 'geo' else manifest['rfe_numeric'] if family == 'rfe' else manifest['curves'])
    categorical = ['curve_suite_id'] if family == 'geo' else []
    permission = variant.get('metadata', 'all')
    numeric = [c for c in numeric if c not in META_NUMERIC]
    if permission in ['all', 'spatial']:
        numeric += META_NUMERIC
    if permission in ['all', 'strata']:
        categorical += STRATA
    if variant.get('no_normalization'):
        numeric = [c for c in numeric if not c.endswith('_well_robust_z')]
    if variant.get('no_context'):
        numeric = [c for c in numeric if not c.endswith('_grad') and not any(f'_w{r}_' in c for r in [2, 5, 10])]
    if variant.get('no_missing'):
        numeric = [c for c in numeric if not c.endswith('_is_missing') and c not in ['missing_count', 'missing_rate']]
        # Retain identical outlier encoding; remove only the explicit missing sentinel.
        numeric = [c[:-4] + '_observed_enc' if c.endswith('_enc') else c for c in numeric]
        categorical = [c for c in categorical if c != 'curve_suite_id']
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


@dataclass
class TrainingFeatureState:
    numeric: list
    categorical: list
    categories: dict
    center: list

    @classmethod
    def fit(cls, train, numeric, categorical):
        categories = {c: sorted(train[c].astype(str).unique().tolist()) for c in categorical}
        positions = train.groupby('well_id', sort=False)[COORDS[:2]].median()
        center = positions.median().to_numpy(float).tolist()
        return cls(list(numeric), list(categorical), categories, center)

    def transform(self, frame):
        names = self.numeric + self.categorical
        out = np.empty((len(frame), len(names)), dtype=np.float32)
        for i, col in enumerate(self.numeric):
            if col == 'dist_to_xy_centroid':
                out[:, i] = np.hypot(frame['x_loc'].to_numpy() - self.center[0], frame['y_loc'].to_numpy() - self.center[1])
            else:
                out[:, i] = frame[col].to_numpy(dtype=np.float32)
        for i, col in enumerate(self.categorical, len(self.numeric)):
            out[:, i] = pd.Categorical(frame[col].astype(str), categories=self.categories[col]).codes
        out[~np.isfinite(out)] = np.nan
        return out

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')
