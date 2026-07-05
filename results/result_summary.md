# GeoMERIT 10-Fold Result Summary

Run: `full10_fullfeat_3gbdt_c2f_tail`

## Setting

- Split: 10-fold GroupKFold by well
- Features: full feature table
- Models: LightGBM + XGBoost + CatBoost ensemble
- Added components: coarse-to-fine classifier, tail experts
- Decode: targeted Bayes-risk penalty decoding
- Best decode parameters:
  - `tau = 0.2`
  - `global_gamma = 1.2`
  - `global_theta = 0.35`
  - `dol_gamma = 8.0`
  - `dol_theta = 0.05`

## Main Metrics

| Metric | Value |
|---|---:|
| Weighted F1 | 0.752551 |
| Macro F1 | 0.531475 |
| Boundary F1 | 0.456735 |
| Penalty | -0.588618 |
| Tail mean F1 | 0.506138 |

## Tail Classes

| Class | F1 |
|---|---:|
| Tuff | 0.700119 |
| Halite | 0.975379 |
| Coal | 0.669925 |
| Dolomite | 0.040099 |
| Anhydrite | 0.651306 |
| Basement | 0.000000 |

## Notes

- Compared with the quick local 3-fold LightGBM run, Penalty improved from about `-0.682` to `-0.589`.
- Compared with the RFE reference values supplied in the design notes, this run is above `Weighted F1 0.727`, above `Boundary F1 0.410`, and better than `Penalty -0.628`.
- Basement remains `0.0` in strict GroupKFold because all 141 Basement samples are absent from the training side in the fold where they appear. Under this protocol the model has no supervised Basement examples for that held-out well.

## Local Files

- `decode_report.json`: full metrics and per-class report
- `decode_report.csv`: per-row OOF predictions
- `decode_grid_results.json`: decode grid search records
- `decode_targeted.log`: decode search log
- `train_full10_fullfeat_3gbdt_c2f_tail.log`: training log
- `run.json`: fold metadata
