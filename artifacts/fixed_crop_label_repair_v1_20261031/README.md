# Source label-time repair v1 — actual repaired data

Completed 2026-10-31. Version: `rflymad_online_binary_v2_labeltime_v1`.

This directory contains repaired 26-channel source train/val CSVs and actual PyTorch caches. It is not a proposal or merely a label overlay. Original directories, source clocks, official crop boundaries, source case splits and retained input values were not overwritten.

## Use these outputs

- `train.pt`: 168550 binary windows, including the original 3680 RflySim normal suffix windows.
- `val.pt`: 35427 binary windows.
- `norm_stats.json`: same historical train-fitted numeric normalization parameters; new dataset version metadata.
- `train_window_links.csv`, `val_window_links.csv`: repaired labels/phase/origin metadata and exact old-cache index mapping.
- `train_source_cache_indices.npy`, `val_source_cache_indices.npy`: each new row maps to its immutable old row.
- `train_excluded_window_ledger.csv`, `val_excluded_window_ledger.csv`: 250 train / 93 val rows held out, without deleting original cases or logs.
- `train_historical_y_same_indices.pt`, `val_historical_y_same_indices.pt`: original labels on precisely the same retained rows, for a future explicitly configured controlled comparison. Not the corrected targets.
- `cases/SIL`, `cases/HIL`: 3135 repaired source CSV copies.
- `case_manifest_source_train_val.csv`, `case_repair_results.jsonl`: case provenance, original/new hashes and row-level changes.
- `repair_manifest.json`: completion status, hashes, limitations and exact counts. Absence of this completion manifest or presence of FAILED.json means the output is incomplete.

## What was changed

The copied CSV `UAVState_data_fault_state` is replaced by the audited original GTD recorded-state reference aligned by timestamp, without moving the official crop. Two explicit flags were added: `labeltime_supervision_valid` and `labeltime_input_clock_review`. All other original CSV field strings are verified unchanged. In total 177618 per-timestamp state entries differ from the original processing.

The compact caches change 3149 retained train targets and 631 retained val targets (3780 total). Unknown endpoint-label rows (122 train / 29 val) and known original inertial-clock review rows (128 train / 64 val) are not materialized into train/val cache tensors, so -1 never reaches binary losses or BatchNorm. The old full data and excluded indices remain available. No replacement windows are drawn and no balance adjustment is applied.

Input X is verified element-for-element equal to the original selected rows. Newly saved targets contain only 0 or 1. The existing `OnlineWindowDataset` class successfully loads, normalizes and yields first/middle/last samples from both caches; no model or optimizer was run.

## Necessary boundaries

- This fixes the established source training/validation label-time error, not every data quality limitation. Existing interpolation and remaining gap flags are retained for an explicitly retrospective source-development comparison; no strict-online guarantee.
- Remaining gap flags occur in 1877 retained train / 389 retained val windows; one old train gap window was already held out for label disagreement. The 0.1s flag is descriptive, not a newly approved exclusion threshold.
- Other label-only command fields, including `UAVState_data_cmd`, remain historical and must NOT be used as new time-correct supervision.
- Corrected CSV labels alone are insufficient: honor both supervision-validity and input-clock flags. The old boolean-only `_infer_fault_mask` now rejects repaired CSVs instead of silently discarding these flags. Use this version's window links and compact caches.
- No test, external_test, Real, open-set or deployment files were modified. Do not mix old final-test labels into a claim of repaired-version performance.
- Old 31-channel deployment models and frozen experiment protocols are unchanged. The general old training CLI expects a test cache; the frozen six-job source runner pins old hashes. Neither should be made to accept this directory by disabling its checks. Configure a new source-only experiment when training is requested.
- The numerical normalization is intentionally fixed for controlled comparison. Some held-out historical training inputs may have contributed to those prior train-only statistics; no claim of retrospective statistical removal.
- No training occurred and no model performance improvement is claimed.

## Reproducibility

Repair entry: `scripts/data/repair_fixed_crop_labeltime.py` in `D:/UAV-Fault-Diagnosis`. A new output directory is required. It verifies original CSV, raw GTD/ULog and audited references before writing, and validates all retained tensors and old input hashes after completion. It does not invoke the original converter or alter source directories.

An initial command timeout occurred after 2092 verified CSVs. Already verified outputs were checked and preserved; one unrecorded partial CSV was saved in `interrupted_files/`, then only the unfinished stage resumed. History is in `RUNNING.json` (initial record) and `RESUME.json`; `repair_manifest.json` is the final completion authority.

Independent verification and execution history: `D:/UAV-Fault-Diagnosis/artifacts/fixed_crop_label_repair_v1_20261031/`.
