# Experiment source and provenance

This directory preserves the final 80-fold experiment's computation and the available small records. The effective scope is in `effective_scope.json`. The original configuration has additional legacy variants; pass the explicit variants in the commands below. The historical scope JSON files are retained together with the later logs-only scope amendment.

The training, feature and decision computations are preserved. Eleven core modules, nested train/report entry points, data configuration and both fixed-parameter training entry points remain byte-identical to their hashes in the final artifact inventory. The dataset loader additionally accepts a compact cohort reference as described below. The revision YAML is the archived original configuration; the frozen protocol also stores its interpreted numerical configuration. The source/publication hash mapping is in `provenance/source_hashes.json`.

The main-report helper and the two fast report files have only root-path portability edits. No numerical parameters, inference definitions, training logic or decision logic were changed. The dataset-reference portability change also preserves the original well folds and per-well class-count checks. The fast report files match the current hashes observed by the 2026-11-16 server review; a contemporaneous manifest was not available to prove their exact bytes at the original report-generation time.

`provenance/records` contains 70 actual selection records (full, RFE and five removals), recorded inner/outer well partitions, protocol definitions and the dataset manifest. It is reference material, not a resumable run. Original logs-only selection files were absent from this local archive; `provenance/logs_only_parameter_mapping.json` identifies the real full-fold parameters inherited by logs-only. Completion markers and model/prediction files are not shipped.

Private host identity and absolute machine paths have been removed from published record copies. Relative input filenames and scientific values remain. Original bytes and publication hashes are distinguished. The historical verifier baseline is a projection containing only the two completion sets that the verifier checks; it contains no process or monitoring logs.

## Inputs and environment

Use Linux with Python 3.12.3. `environment/constraints-recorded.txt` pins the eight recorded package versions; it is not a complete environment lock. Install with `python -m pip install -r requirements-runtime.txt`. Other I/O and transitive package versions were not recorded. No claim of an independently rerun end-to-end reproduction is made.

Acquire the public FORCE 2020 LAS data and auxiliary inputs separately. Retain these paths relative to this directory:

- `source/Force_2020_all_wells_train_test_blind_hidden_final/`
- `source/lithology scoring matrix cost function.xlsx`
- `source/NPD_Casing_depth_most_wells.xlsx`
- `source/NPD_Lithostratigraphy_groups_all_wells.xlsx`
- `source/NPD_Lithostratigraphy_member_formations_all_wells.xlsx`

The original data configuration already uses these relative paths. Raw data, feature caches, fitted models, probabilities, point predictions and virtual environments are not included. Do not copy provenance completion records into fresh run directories.

## Explicit training and reporting sequence

Run from this experiment directory; these commands train models and are provided for reproduction, not run during publication. A suitable CPU host and substantial time, RAM and free disk are required.

```sh
python source/geomerit/scripts/10_revision_train.py --work experiment_v1 --variants full rfe_xgb --folds 0 1 2 3 4 5 6 7 8 9
mkdir -p experiment_fixed_parameters_20261104 experiment_fixed_parameters_logs_only_20261110
ln -s ../experiment_v1/dataset experiment_fixed_parameters_20261104/dataset
ln -s ../experiment_v1/dataset experiment_fixed_parameters_logs_only_20261110/dataset
python fixed_ablation_train.py --work experiment_fixed_parameters_20261104 --variants no_missing no_normalization no_class_weight no_boundary_weight no_confidence_weight --folds 0 1 2 3 4 5 6 7 8 9
python fixed_logs_only_train.py --init
python fixed_logs_only_train.py --work experiment_fixed_parameters_logs_only_20261110 --variants logs_only --folds 0 1 2 3 4 5 6 7 8 9
python parallel_execution/fixed_main_report.py --work experiment_v1
python parallel_execution/report_tools_fast/final_geomerit_report.py --output reports/final_all_80
```

Report roots default to the location of this package. `GEOMERIT_REVISION_ROOT` can explicitly point the report scripts to a complete compatible experiment tree. The final report verifier requires all actual prediction/model/checkpoint artifacts and the recorded package versions; published summary tables alone cannot satisfy it. The old controller source files are retained only because the verifier inventories their hashes. They are not the default execution route, and no service installation, monitoring or remote-connection tooling is included.

The original non-backend protocol tests can be selected with unittest. The original test named `test_noncontiguous_class_mapping_all_backends` performs tiny backend fits, so it is excluded from publication-time validation. Full protocol tests require the listed research dependencies.

## Compact cohort reference for a fresh checkout

The original dataset loader read two earlier prediction-reference files only to recover the well-fold map and per-well class counts. `source/geomerit/configs/cohort_reference.json` contains their exact aggregation, with both original SHA256 values. The publication loader uses it when both legacy files are absent and records its hash in each newly built dataset manifest. The historical references and compact reference were checked to produce identical maps, counts and the 1,431,383-row total across all 118 wells. The original per-well class-count assertions still apply to data rebuilt from LAS; feature construction, labels and model computations are unchanged. A fresh checkout therefore needs the public raw inputs, not the earlier prediction files.

Published code hashes distinguish these portability edits from the original experiment. A newly executed run records its own protocol signature; existing completion markers must not be mixed with newly generated outputs. Historical README/protocol text is explicitly labeled; use the commands above and `effective_scope.json` for this release.
