# Result Files

This folder keeps lightweight result summaries that are suitable for GitHub.

- `result_summary.md`: human-readable summary of the main 10-fold run.
- `decode_report.json`: pooled metrics and per-class scores.
- `decode_grid_results.json`: penalty-decoding grid-search records.

The full OOF prediction CSV, probability arrays, fold-level model outputs, and
retrieval/stacking tensors are intentionally excluded because they are large and
can be regenerated from the scripts.

