#! /bin/bash
# Launch OCTCube-IR dual-modality (3D OCT + 2D en-face IR) fine-tune/eval for all
# main-benchmark UF-cohort studies, one run per study on the full training set
# (SUBSETNUM=0), matching the single-run protocol of finetune_UFcohort_IRB2024v5.sh.
#
# Both towers are initialized from the single OCTCube-IR joint checkpoint
# (mm_octcube_ir.pt). Study / class-count / task_mode combos mirror
# UFcohort_multi_dualeval.sh.

SCRIPT="scripts/finetune_UFcohort_IRB2024v5_OCTCubeIR.sh"

# Optional inputs (with defaults matching the single-run benchmark protocol):
#   bash scripts/UFcohort_multirun_OCTCubeIR.sh [Eval_score] [SUBSETNUM] [MODEL]
# e.g. bash scripts/UFcohort_multirun_OCTCubeIR.sh AUPRC 0
Eval_score=${1:-"AUPRC"}
SUBSETNUM=${2:-0} # 0 (full training set), 500, 1000
MODEL=${3:-"flash_attn_vit_large_patch16"}

echo "Eval_score=$Eval_score  SUBSETNUM=$SUBSETNUM  MODEL=$MODEL"

# study                        num_class  task_mode
sbatch $SCRIPT AMD_all_split               $MODEL 2 $Eval_score binary_cls $SUBSETNUM
sbatch $SCRIPT Cataract_all_split          $MODEL 2 $Eval_score binary_cls $SUBSETNUM
sbatch $SCRIPT DR_all_split                $MODEL 6 $Eval_score multi_cls  $SUBSETNUM
sbatch $SCRIPT Glaucoma_all_split          $MODEL 6 $Eval_score multi_cls  $SUBSETNUM
sbatch $SCRIPT DR_binary_all_split         $MODEL 2 $Eval_score binary_cls $SUBSETNUM
sbatch $SCRIPT Glaucoma_binary_all_split   $MODEL 2 $Eval_score binary_cls $SUBSETNUM
