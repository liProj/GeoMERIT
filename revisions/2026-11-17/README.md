# GeoMERIT revision experiments — 2026-11-17

[中文说明](README.zh-CN.md)

This dated snapshot publishes the completed revision experiments summarized on
2026-11-16 and the accompanying missingness-stratum and error analyses updated
through 2026-11-17. It contains **80 completed outer-fold runs**, with eight
configurations evaluated on the same **118 wells and 1,431,383 labeled samples**.
The repository's original source and earlier results remain outside this
directory.

## Contents and entry points

| Directory | Contents |
| --- | --- |
| [experiment/](experiment/PROVENANCE.md) | Revision training and reporting source, configuration, protocol tests, recorded well splits and selected parameters, and the effective experiment scope. |
| [results/](results/README.md) | Final pooled and per-fold results, class support, paired comparisons, adjusted intervals, parameter tables, prediction provenance and verification records. |
| [analysis/](analysis/README.md) | Runnable four-stratum whole-well bootstrap analysis and S7 class-wise error decomposition. |

Start with the [experiment instructions](experiment/PROVENANCE.md) for environment,
data layout, dataset construction, nested main comparisons, and fixed-setting
retraining. The [analysis instructions](analysis/README.md) explain how to
recompute the supplementary analyses after predictions and reports exist.
The [results index](results/README.md) maps each published data file to its role.

## Effective experiment design

- The two main configurations, `full` and `rfe_xgb`, use ten outer well-grouped
  folds, with three inner well-grouped folds for their specified parameter
  searches. Every well is evaluated in its outer holdout fold.
- The six retrained removals are `no_missing`, `no_normalization`,
  `no_class_weight`, `no_boundary_weight`, `no_confidence_weight`, and `logs_only`.
  Each inherits the fusion and decision parameters selected by the full model
  within the corresponding outer training fold, holding these settings fixed
  while refitting the outer models. Training-dependent feature state, priors,
  weights, and expert eligibility are recomputed for the variant.
- `logs_only` removes geological, spatial, and casing metadata and retains the
  log-derived representation. Its ten completed outer fits use the same-fold
  full-model parameter inheritance, rather than a separate search. The archive
  provides the original full-model parameter records and the inheritance mapping;
  the mapping documents the inherited settings for all ten logs-only folds.
- The effective scope is eight configurations × ten folds, totaling **80 outer
  fits and 140 model-fitting blocks**. The experiment instructions distinguish
  this final scope from historical protocol and scheduling records.

## Pooled held-out results

| Output | Weighted F1 | Macro F1 | Boundary F1 | Penalty |
| --- | ---: | ---: | ---: | ---: |
| GeoMERIT full | 0.7495 | 0.5286 | 0.4618 | -0.6157 |
| RFE argmax | 0.7155 | 0.4796 | 0.4418 | -0.7917 |
| RFE adjusted + risk decoding | 0.7151 | 0.4819 | 0.4418 | -0.7848 |
| no_missing | 0.7203 | 0.4786 | 0.4436 | -0.7284 |
| no_normalization | 0.6956 | 0.4462 | 0.4369 | -0.8245 |
| no_class_weight | 0.6983 | 0.4152 | 0.4374 | -0.8088 |
| no_boundary_weight | 0.7185 | 0.4657 | 0.4402 | -0.7326 |
| no_confidence_weight | 0.7231 | 0.4900 | 0.4530 | -0.7079 |
| logs_only | 0.7039 | 0.4161 | 0.4370 | -0.7543 |

Penalty is the negative average geological misclassification cost; higher is
better. Both RFE rows are outputs of the same ten `rfe_xgb` outer fits, not
additional trained configurations. The CSV and JSON files retain the unrounded
values; rounded displayed scores can differ from rounded paired differences.

Relative to `logs_only`, the full model improves Weighted F1 by **0.0457**,
Boundary F1 by **0.0248**, and Penalty by **0.1386**. The corresponding
Bonferroni-adjusted whole-well bootstrap intervals are **[0.0267, 0.0674]**,
**[0.0119, 0.0382]**, and **[0.0694, 0.2199]**.

The six retrained ablations and three primary metrics form an **18-comparison
family**. Seventeen adjusted intervals are positive; the interval for removing
confidence weighting on Boundary F1 crosses zero. The main RFE comparisons use
their separate six-comparison family. Paired whole-well resampling uses 5,000
draws and is conditional on the fitted predictions; Holm-adjusted fold-level
tests are reported separately as secondary analyses. All directions and
intervals, including the interval crossing zero, are retained in `results/`.

## Supplementary analyses

The four disjoint well-missingness strata contain 55, 40, 18, and 5 wells and
cover 12, 11, 11, and 8 lithology classes. Their Weighted F1 values are 0.7664,
0.7737, 0.6572, and 0.7453. The Sandstone/Shale mixed class accounts for 21.21%
of the `[0.5,0.6)` stratum; Shale and Sandstone together account for 80.81% of
the highest-missingness stratum. Each within-stratum bootstrap restarts seed
20261103 and uses 5,000 whole-well draws. These are observed subgroup results,
not artificial masking experiments.

S7 decomposes the conditional single-XGBoost/full-model trade-off and all
Dolomite prediction destinations. Of the 2,352 Dolomite errors, 945 are assigned
to Shale and 933 to Limestone, together 79.85%. The analysis reports observed
error allocation without assigning an untested physical cause.

## Data, provenance, and reproduction scope

Download the public FORCE 2020 dataset and required auxiliary files from the
[Zenodo record](https://doi.org/10.5281/zenodo.4351156), and place them according
to the experiment configuration. Raw LAS/Excel data, feature caches, fitted
models, probability arrays, and per-row predictions are not distributed here.
They are generated or supplied locally for the training and analysis commands.

Published reports retain the artifact hashes, fold identifiers, exact metrics,
and public-relative artifact locations. Administrative machine paths and stale
progress fields are normalized or removed in the publication copies; their
transformations and source hashes are documented with the relevant material.
The experiment archive distinguishes original recorded parameters from the
logs-only inheritance mapping, and records the known package versions without
claiming to provide a complete environment lock.

Local publication checks cover source/protocol checks, result consistency,
analysis recomputation, and the snapshot file manifest. A new full 80-fold
training run was not performed for the publication step. Hardware and library
differences can affect a new training run, so compare regenerated scores with
their own prediction and report provenance rather than assuming byte-identical
model outputs.

## Verify the downloaded snapshot

Run `python verify_snapshot.py` from this directory (Python standard library only).
It checks every file listed in `MANIFEST.sha256` against its published bytes.
This verifies the downloaded publication files; the experiment artifact verifier
is a separate check for regenerated model and prediction outputs.
