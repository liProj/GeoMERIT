# Results released on 2026-11-17

This directory contains the completed 80-fold GeoMERIT results and subsequent descriptive analyses used in the manuscript. Release date is distinct from the execution and verification timestamps retained in the reports. All original metric values and interval endpoints are preserved; CSV copies retain their original numeric precision.

## Files

| File | Content |
|---|---|
| `final_report.json` | Final nine model/output summaries, 18 ablation contrasts, and 80 prediction provenance entries |
| `main_report.json` | Main and conditional outputs, six main comparisons, per-class results and missingness results |
| `artifact_verification.json` | Completed folds, 140 training blocks, prediction checks and artifact SHA256 inventory; machine-monitoring fields removed |
| `summary_metrics.csv` | Nine final outputs, pooled metrics |
| `fold_metrics.csv` | Nine outputs by ten folds (90 rows); output count differs from the eight trained variants because RFE exports two decisions |
| `pooled_metrics.csv` | All 24 reported trained/conditional outputs and six metrics |
| `main_comparisons.csv` | Six paired GeoMERIT-minus-RFE comparisons |
| `ablation_comparisons.csv` | All 18 paired GeoMERIT-minus-removal comparisons |
| `fold_mean_sd.csv` | Arithmetic fold means and sample SDs for 24 outputs and six metrics |
| `per_class.csv` | Full-model precision, recall, F1 and support for all 12 lithologies |
| `selected_parameters_full.csv` | Actual full-model blend and decision selections, checked against ten parameter files in the final SHA256 inventory |
| `selected_parameters_rfe.csv` | RFE selections from ten archived records, paired with completion markers referencing the final prediction digests |
| `missingness_summary.csv` | Four disjoint natural-missingness strata with within-stratum intervals |
| `missingness_class_support.csv` | Class-by-stratum sample and well counts |
| `error_mechanism_S7.csv` | Original-precision class contribution and Dolomite prediction-destination analyses; original Chinese column labels retained |
| `source_sha256.json` | Source digests identified by logical, non-private source names |

The archived main report's administrative `pending` field predates completion of the remaining ablations. It is omitted in this release because the final report supersedes that historical status and records all 80 completed folds. The numerical main-comparison results are unchanged.

## Actual evaluation protocol

The evaluation covers 118 wells, 1,431,383 labelled samples and 12 classes. `full` and `rfe_xgb` each use ten outer well-held-out folds and three inner well-grouped folds for their specified parameter searches. The five `no_*` removals and `logs_only` each refit ten outer models while inheriting the full model's same-fold blend and decision selections. Thus eight variants produce 80 outer folds and 140 training blocks. Logs-only is a fixed-setting metadata-removal experiment, not an independently tuned model. It removes coordinates, training-center distance, casing features, GROUP and FORMATION jointly while retaining log-derived feature engineering.

Full-model blend selection considers two backend-weight vectors and 11 hierarchy coefficients; the subsequent decision search evaluates 500 candidates. The CSV records are the executed fold selections, not recommended universal defaults. RFE exports both argmax and its selected adjusted/risk decision. In the RFE parameter CSV, `source=single fixed backbone` identifies the single XGBoost model; alpha/mu retain the common export schema. All six removal variants reuse the corresponding full-model selections; the report provenance records their separate outer refits. The 24 pooled outputs also include descriptive conditional branch/backend/decoder outputs from already fitted models, not 24 independently trained variants.

Whole-well paired bootstrap intervals use 5,000 resamples and seed 20261103, conditional on the fitted models and observed wells. The main family has six outcomes; the ablation family has 18. Bonferroni simultaneous intervals use the corresponding family size. Fold Wilcoxon tests with Holm adjustment are secondary because outer training sets overlap. Fold SD uses denominator n-1 and is distinct from pooled-prediction metrics. Boundary F1 is weighted lithology F1 on marked boundary samples, not boundary-event detection F1. Penalty is negative mean geological cost; higher is better.

Natural-missingness strata are [0,0.4), [0.4,0.5), [0.5,0.6) and [0.6,1], based on the same 20 logging curves. They contain 55/40/18/5 wells and 12/11/11/8 classes. Their intervals and nonmonotonic pooled scores are descriptive and do not identify a causal missingness effect. No spatial-block/field-held-out or artificial-missingness experiment is represented here. S7 is a decomposition of saved held-out predictions and does not identify a physical cause of errors.

The canonical four disjoint strata are in `missingness_summary.csv` and `missingness_class_support.csv`. The raw `main_report.json` additionally retains two overlapping threshold aggregates. Its upper endpoint 1.000001 is the implementation's exclusive cutoff for including missingness equal to 1; original report values are preserved.

## Artifact paths and provenance

Absolute private prefixes have been removed from report paths. Paths beginning with `experiment_v1/`, `experiment_fixed_parameters_20261104/` or `experiment_fixed_parameters_logs_only_20261110/` refer to the expected external artifact-root layout. They match the relative entries in `artifact_verification.json`; they are not assertions that the large prediction/model artifacts are included in this Git repository. Prediction SHA256 values, variants and fold identifiers are retained. `main_report_path` resolves within this directory. Runtime process/CPU/free-memory/free-disk fields are omitted. The original report digests are retained in `source_sha256.json` for provenance.

No reviewer text, response letters, private machine paths, credentials, monitoring logs, raw training data, model binaries or per-sample prediction files are included in this directory.
