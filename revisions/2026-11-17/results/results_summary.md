# Completed-result summary

Pooled outer-held-out metrics are shown below; full precision is retained in the machine-readable files.

| Output | Weighted F1 | Macro F1 | Tail F1 | Boundary F1 | Accuracy | Penalty |
|---|---:|---:|---:|---:|---:|---:|
| full/full | 0.749521 | 0.528556 | 0.500261 | 0.461826 | 0.758149 | -0.615716 |
| rfe_xgb/posterior_argmax | 0.715476 | 0.479645 | 0.390343 | 0.441832 | 0.702014 | -0.791656 |
| rfe_xgb/full | 0.715089 | 0.481900 | 0.394161 | 0.441794 | 0.697919 | -0.784778 |
| logs_only/full | 0.703862 | 0.416083 | 0.351864 | 0.437001 | 0.712915 | -0.754299 |

GeoMERIT Weighted F1 is **0.7495214811997727**; logs-only is **0.7038621814124536**. The full-minus-logs-only difference is **0.045659299787319085**. Logs-only inherits the full model's same-fold selected blend and decision settings.

| Full minus logs-only | Difference | Bonferroni simultaneous interval |
|---|---:|---|
| weighted_f1 | 0.045659300 | [0.026699390, 0.067425278] |
| boundary_f1 | 0.024825131 | [0.011890628, 0.038195072] |
| penalty | 0.138582755 | [0.069448893, 0.219854925] |

All six main-comparison intervals are positive. Of the 18 ablation intervals, **17 are positive and one crosses zero**: the no-confidence-weight Boundary F1 contrast, with difference 0.008782048195388747 and interval [-0.0031469589060317653, 0.021162236089162685]. All six ablation Weighted F1 intervals are positive. All three logs-only intervals are positive; this does not imply that every secondary Holm-adjusted fold test is significant.

The full output improves Weighted, Macro and Boundary F1 over single-XGBoost argmax, while XGBoost has higher Accuracy and Penalty. The full output is not the highest Boundary F1 among every conditional branch output. The S7 decomposition records XGBoost's higher share of Shale predictions and its net gains on true Shale partly offset by net losses on other classes. Dolomite has 2,352 errors, of which 945 are assigned to Shale and 933 to Limestone (79.8469387755102% together); the 39 correct Dolomite predictions are excluded from that error percentage.

In the [0.5,0.6) missingness stratum, Sandstone/Shale accounts for 51,919/244,825 samples (21.21%). In [0.6,1], Shale and Sandstone together account for 67,089/83,016 samples (80.81%). Composition differences coexist with nonmonotonic performance; they do not establish a root cause.

The no-missingness-encoding Weighted F1 difference is 0.029233292231311436 with interval [0.006463964926523823, 0.052799125550916794]. The no-robust-normalization difference is 0.053872240513505365 with interval [0.025629051904838728, 0.09016540300833308]. Both are fixed-setting outer-refit comparisons.
