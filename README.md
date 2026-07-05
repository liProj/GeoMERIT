# GeoMERIT

GeoMERIT is a CPU-friendly lithology prediction pipeline for real well-log data with non-random missingness, inter-well acquisition heterogeneity, long-tailed lithology labels, boundary ambiguity, and asymmetric geological error costs.

The code accompanies the manuscript:

**GeoMERIT: Missingness-Aware and Penalty-Guided Robust Lithology Prediction from Well Logs**

## What This Repository Contains

This repository is a paper-reference release. It contains source code, model configuration files, plotting scripts, lightweight result summaries, and the LaTeX manuscript assets needed to understand and reproduce the reported experiments. Large raw datasets, out-of-fold probability arrays, model caches, and full per-row prediction tables are intentionally excluded from GitHub.

```text
.
|-- geomerit/              # Core Python package
|   |-- io_las.py          # LAS and auxiliary Excel readers
|   |-- features.py        # Missingness-aware robust feature engineering
|   |-- labels.py          # FORCE 2020 lithology labels and class groupings
|   |-- weights.py         # Class, boundary, and confidence sample weights
|   |-- models.py          # GBDT ensemble, hierarchy, and tail experts
|   |-- decode.py          # Logit adjustment and Bayes-risk decoding
|   |-- metrics.py         # Weighted F1, Macro F1, Boundary F1, Penalty
|   `-- cv.py              # Well-grouped cross-validation helpers
|-- scripts/               # Reproduction entry points
|   |-- 00_build_dataset.py
|   |-- 01_train.py
|   |-- 02_predict_decode.py
|   |-- 03_ablation.py
|   |-- 04_georacs_oof.py
|   |-- 04_make_figures.py
|   `-- 05_make_paper_figures.py
|-- configs/               # Dataset, feature, model, and penalty configs
|-- results/               # Lightweight experiment reports
|-- paper/                 # Latest manuscript, figures, and figure scripts
|   |-- latex/
|   |-- figures/
|   |-- scripts/
|   `-- data/
`-- requirements.txt
```

## Method Summary

GeoMERIT is built around four robustness layers:

1. **Missingness-aware representation**: missing curves, anomalous values, within-well robust normalization, depth-window context, and curve-suite identity are encoded explicitly instead of being imputed away.
2. **Long-tail and boundary-aware learning**: class weights, boundary weights, and label-confidence weights reshape the training objective toward rare lithologies and stratigraphic transitions.
3. **Structured posterior modeling**: LightGBM, XGBoost, and CatBoost posteriors are fused with a coarse-to-fine geological hierarchy and one-vs-rest tail experts.
4. **Penalty-guided decision layer**: final lithology labels are selected by logit adjustment and Bayes-risk decoding over the FORCE 2020 geological penalty matrix, rather than plain posterior argmax.

## Reported 10-Fold Results

Strict 10-fold GroupKFold by well on the FORCE 2020 valid-label set:

| Metric | Value |
|---|---:|
| Weighted F1 | 0.752551 |
| Macro F1 | 0.531475 |
| Boundary F1 | 0.456735 |
| Geological Penalty | -0.588618 |
| Tail mean F1 | 0.506138 |

Tail-class F1:

| Class | F1 |
|---|---:|
| Tuff | 0.700119 |
| Halite | 0.975379 |
| Coal | 0.669925 |
| Dolomite | 0.040099 |
| Anhydrite | 0.651306 |
| Basement | 0.000000 |

Basement remains unresolved under strict held-out-well validation because its 141 labeled samples are concentrated in very few wells, and some GroupKFold training folds contain no supervised Basement examples.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Build the feature table after downloading the FORCE 2020 data and placing the auxiliary NPD Excel files according to `configs/data.yaml`:

```bash
python scripts/00_build_dataset.py \
  --config configs/data.yaml \
  --features configs/features.yaml \
  --out data/feature_table.parquet
```

Train the well-grouped model:

```bash
python scripts/01_train.py \
  --table data/feature_table.parquet \
  --split groupkfold \
  --folds 10 \
  --group well_id \
  --config configs/model_full10.yaml \
  --components flat,coarse2fine,tail_experts \
  --out runs/full10_fullfeat_3gbdt_c2f_tail
```

Decode and evaluate:

```bash
python scripts/02_predict_decode.py \
  --run runs/full10_fullfeat_3gbdt_c2f_tail \
  --eval oof \
  --penalty configs/penalty_matrix.csv \
  --config configs/model_full10.yaml \
  --objective penalty \
  --out runs/full10_fullfeat_3gbdt_c2f_tail/decode_report.json
```

## Data

The FORCE 2020 well-log and lithofacies dataset is publicly available from Zenodo:

https://doi.org/10.5281/zenodo.4351156

Raw LAS files, large feature tables, model arrays, and full per-row OOF prediction tables are not included in this repository. This keeps the repository small and avoids redistributing the benchmark data.

## Paper Figures

The latest paper figures and plotting scripts are under `paper/`:

```bash
cd paper/scripts
python make_figures.py
```

The `paper/data/` folder contains the small CSV files needed for representative well-track figures in the manuscript. Other figure data are derived from the reported result summaries and public FORCE 2020 metadata.

## Citation

If this code supports your research, please cite the accompanying manuscript:

```bibtex
@article{geomerit2026,
  title   = {GeoMERIT: Missingness-Aware and Penalty-Guided Robust Lithology Prediction from Well Logs},
  author  = {Anonymous Author},
  journal = {Big Data and Cognitive Computing},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Reproducibility Notes

- Validation is always grouped by well; random row-wise splits are not used for headline results because they leak local depth context and well-specific acquisition signatures.
- The main experiments are CPU-only. The reported full 10-fold run used a 112-core CPU server with approximately 251 GB RAM.
- The repository contains lightweight result summaries in `results/`. Full OOF predictions and probability tensors can be regenerated with the scripts above.
