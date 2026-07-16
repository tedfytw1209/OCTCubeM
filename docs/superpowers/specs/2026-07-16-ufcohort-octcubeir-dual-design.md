# UF Cohort fine-tuning & evaluation on OCTCube-IR (dual modality) — design

Date: 2026-07-16

## Motivation (reviewer comment)

A reviewer asked the manuscript to state, per experiment, whether **OCTCube** (3D
OCT-volume-pretrained, structural OCT only) or **OCTCube-IR** (3D OCT volume **+**
2D en-face IR image, jointly pretrained) is used, because the two are different
comparison strengths and must not be described interchangeably. To back the claim
of "consistently surpassing existing single-modal and multimodal foundation
models", the most relevant multimodal variant (OCTCube-IR) must be evaluated on
the UF cohort under a comparable **3D OCT + 2D en-face** setting.

This work adds a dedicated UF-cohort fine-tune/eval entry point whose **both**
encoder towers are initialized from the single jointly-pretrained OCTCube-IR
checkpoint, so the result is unambiguously an "OCTCube-IR" number.

## What already exists (and why it is not sufficient)

- `main_finetune_downstream_UFcohort.py` — the canonical **single-modality**
  OCTCube benchmark script (3D OCT only). The dataset construction, subset/
  bootstrap logic, training loop, and evaluation protocol here are the reference
  we mirror.
- `main_finetune_downstream_UFcohort_dual.py` + `engine_finetune_dual.py` — an
  existing dual pipeline. It initializes the **OCT** tower from plain
  `OCTCube.pth` and only the **en-face** tower from `mm_octcube_ir.pt` (its
  checkpoint loader keeps `text.*` keys only). That is a *hybrid* init, not a
  model whose both towers trace back to OCTCube-IR joint pretraining — exactly
  the ambiguity the reviewer flagged. Its `--not_print_logits`-masked
  `samples.shape` NameError also makes it fragile.
- `retinal-COEM/src/open_clip/` — defines the OCTCube-IR CLIP model whose
  `state_dict` uses `visual.*` (3D OCT tower) and `text.*` (2D en-face tower)
  key prefixes. `mm_octcube_ir.pt` is a checkpoint of that model. The OCT tower
  is partially unfrozen during joint pretraining (`--lock-image-unlocked-groups
  9`), so it genuinely differs from vanilla `OCTCube.pth`.

## Design

New file **`OCTCube/main_finetune_downstream_UFcohort_OCTCubeIR.py`**, structured
after `main_finetune_downstream_UFcohort.py` (arg parser incl. `--subsetseed`/
`--bootstrap_runs`/`--droplast`, `--seed 42`, bootstrap wandb/`model_add_dir`
handling, single-split branch, best-model reload → final test), adapted for dual
modality (from `main_finetune_downstream_UFcohort_dual.py`).

### Model + checkpoint loading — follow retinal-COEM (the OCTCube-IR model)
The model reproduces retinal-COEM's `CustomTextCLIPClassification`
(`retinal-COEM/src/open_clip/model.py:741`), NOT OCTCube's `DualViT`
average-of-logits. `retinal-COEM/src/open_clip` cannot be imported here (its
`model.py` imports a missing `model_backup` package), so the model is replicated
on the OCTCube side using the architecturally-identical OCTCube tower classes
(`models_vit_st_flash_attn_nodrop`, `models_vit_flash_attn`), arranged like
OCTCube's `DualViTClassifier` (a combined `self.blocks`) so it plugs into
`engine_finetune_dual` and `lrd.param_groups_lrd` unchanged.

- Each tower is built with its head projecting to `embed_dim` (=`--mm_embed_dim`,
  default 512 — the OCTCube-IR CLIP embedding dim), i.e. `num_classes=mm_embed_dim`
  plays retinal-COEM's `out_dim=embed_dim` role, so the checkpoint's
  `visual.head`/`text.head` (512×1024) match and are KEPT.
- `forward(oct, cfp)`: `F.normalize(visual(oct))` and `F.normalize(text(cfp))`
  (L2, as in retinal-COEM's `encode_image`/`encode_text`) → concatenate
  (2×embed_dim) → `ClassificationHead` (LayerNorm → Linear → GELU → Linear) →
  `num_classes` logits. Returns logits only (so the `_dual.py` engine, which
  treats the output as logits, works unchanged).

A single `--finetune` points at `mm_octcube_ir.pt`
(`/blue/ruogu.fang/tienyuchang/OCTCubeM/ckpt/mm_octcube_ir.pt`). The loader reads
`checkpoint['state_dict']` (falls back to `['model']`/raw), strips a leading
`module.`, splits into `visual.*` → OCT tower and `text.*` → en-face tower, and
loads each tower with retinal-COEM's own per-tower recipe (`_build_vision_tower`
/`_build_text_tower`): drop `head` only on shape mismatch (here it matches, so it
is kept), interpolate spatial (and temporal, for OCT) position embeddings, load
`strict=False`. The fresh `classification_head` stays randomly initialized.

Guard: assert each split is non-empty (catches a wrong-prefix checkpoint), print
`missing_keys` for both towers. Because `mm_octcube_ir.pt` is not in the repo
checkout, a standalone `inspect_mm_octcube_ir_keys.py` prints the checkpoint's
top-level key-prefix counts so the `visual.`/`text.` assumption can be confirmed
before launching the sweep.

### Dataset / finetune / evaluation — follow `main_finetune_downstream_UFcohort_dual.py`
`Dual_Dataset(PatientDataset3D, PatientDataset2D)` built from the same UF-cohort
CSV; `monai_3D` transforms for OCT, 2D transforms for en-face; layer-wise-decay
AdamW; per-epoch `train_one_epoch_dual`/`evaluate_dual`; select best by
`--val_metric`; on each best-val improvement, save `checkpoint-best.pth` and
evaluate the test set (test metric = test @ best-val epoch, as in `_dual.py`).
`Dual_Dataset`'s `.targets`/`.classes`/`.class_to_idx`/`.annotations` are
forwarded from its OCT sub-dataset so the subset path stays functional.

Two deliberate deviations from `_dual.py`, both correctness fixes: (1) the
train sampler uses `shuffle=True` (`_dual.py` uses `shuffle=False`, which would
freeze the training order across epochs); (2) `misc.load_model` is called only
when `--resume` is set (`_dual.py` calls it unconditionally, and `misc.load_model`
raises on an empty `--resume`).

### Non-goals / out of scope
- Not fixing `retinal-COEM/src/open_clip`'s unrelated import breakage.
- Not touching the existing `_dual.py`/`_dualeval.sh` (kept reproducible).
- Only `engine_finetune_dual.py`'s latent `samples.shape` NameError is fixed in
  place (strict improvement, masked today by `--not_print_logits`).

## Experiment protocol (confirmed with user)
- Both towers initialized from `mm_octcube_ir.pt`.
- All 6 main-benchmark studies: AMD, Cataract, DR (multi + binary), Glaucoma
  (multi + binary), each with its class count / metric / task_mode.
- Single run per study on the full training set (`SUBSETNUM=0`), matching the
  plain `finetune_UFcohort_IRB2024v5.sh` protocol.

## Deliverables
1. `OCTCube/main_finetune_downstream_UFcohort_OCTCubeIR.py`
2. `OCTCube/scripts/finetune_UFcohort_IRB2024v5_OCTCubeIR.sh` (single study)
3. `OCTCube/scripts/UFcohort_multirun_OCTCubeIR.sh` (all 6 studies)
4. `OCTCube/inspect_mm_octcube_ir_keys.py` (pre-flight key check)
5. `engine_finetune_dual.py` `samples.shape` fix; CLAUDE.md filename correction.

Runs on the HiPerGator cluster (`sbatch`); not runnable in this dev environment
(no GPU/data/torch).
