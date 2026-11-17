> Historical source documentation. For the completed 80-fold release, effective scope and current runnable commands, use [the publication instructions](../../PROVENANCE.md). Earlier entry points or proposed variants below are retained only as historical context.

# GeoMERIT

GeoMERIT is a CPU-friendly FORCE 2020 lithology classification pipeline:
missingness-aware feature engineering, GBDT ensembles, long-tail correction,
Bayes risk decoding, and Viterbi stratigraphic smoothing.

## Quick Start

```powershell
cd geomerit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/00_build_dataset.py --config configs/data.yaml --features configs/features.yaml --out data/feature_table.parquet
python scripts/01_train.py --table data/feature_table.parquet --split groupkfold --config configs/model.yaml --out runs/gkf
python scripts/02_predict_decode.py --run runs/gkf --eval oof --penalty configs/penalty_matrix.csv
```

The scripts are written so the RFE-style baseline can be reproduced first, then
additional GeoMERIT components can be enabled or disabled through ablations.
