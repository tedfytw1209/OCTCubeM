# Fundus Validation-Selected Test Protocol Design

## Goal

Make `main_finetune_downstream_public2D_OCTCubeIR_fundus.py` follow the supplied RETFound benchmark protocol: choose one checkpoint using validation data only, then evaluate the test split exactly once from that checkpoint.

## Scope

This change is limited to the OCTCube-IR public fundus fine-tuning entry point and its launcher. It fixes validation selection for `AUC`, `AUPRC`, `ACC`, and `BalAcc`; removes test-based selection; and makes best-checkpoint persistence mandatory for training. It does not change datasets, transforms, optimization, model architecture, or distributed evaluation behavior.

## Selection Semantics

Each epoch produces a validation metric snapshot. The configured `--val_metric` maps to exactly one primary score:

- `AUC` uses `val_auc_roc`.
- `AUPRC` uses `val_auc_pr`.
- `ACC` uses `val_stats['acc1']`.
- `BalAcc` uses `val_bal_acc` and requires `--return_bal_acc`.

An epoch becomes best only when its primary score is strictly greater than the previous best score. Equal scores retain the earlier checkpoint, matching RETFound's `max_score < val_score` behavior. When an epoch becomes best, all reported validation metrics are copied from that same epoch; maxima from different epochs are never combined.

## Checkpoint and Test Data Flow

On each validation improvement, rank zero writes `checkpoint-best.pth` through the existing `misc.save_model` interface. Best-checkpoint saving is required by the training protocol and no longer depends on `--save_model`; the argument remains accepted for command-line compatibility. The fundus launcher will explicitly pass `--save_model` to document this behavior.

After training finishes successfully, all ranks synchronize, load `checkpoint-best.pth` into `model_without_ddp`, and evaluate the test loader exactly once. No test evaluation occurs inside the epoch loop, no test metric participates in selection, and the final test result is the complete metric snapshot from the validation-selected checkpoint.

If no validation epoch completes successfully, the program raises an error instead of writing zero-valued result files. If `--val_metric BalAcc` is requested without `--return_bal_acc`, argument processing raises a clear error before training begins.

## Results and Logging

Existing output names remain stable: validation results go to `results.txt`, test results go to `results_test.txt`, and final W&B keys retain the `final/val_*` and `final/test_*` prefixes. Per-epoch W&B/TensorBoard validation logging remains. Per-epoch test logging and test-max tracking are removed.

`max_val_epoch` identifies the checkpoint used for the single test evaluation. Validation and test dictionaries each represent one model state rather than independently maximized fields.

## Testing

Regression tests will exercise a small, dependency-light selection helper so they do not require CUDA, flash-attention, datasets, or model checkpoints. Tests will prove:

- ACC is mapped and selectable.
- AUC strict improvement replaces the best epoch.
- AUC ties retain the earlier epoch.
- A validation snapshot is internally consistent.
- BalAcc without a computed value is rejected.
- The training source has no test evaluation inside the epoch loop and reloads `checkpoint-best.pth` before its one post-training test evaluation.

The launcher will also be checked for an explicit `--save_model` argument.
