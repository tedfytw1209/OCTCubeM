#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module purge
module load conda
conda activate octcube

SCRIPT=${1:-"scripts/finetune_UFcohort_IRB2024v5_dualeval_OCTCubeIR.sh"}
Eval_score=${2:-"AUPRC"}
SUBSETNUM=${3:-0} # 0 (full training set), 500, 1000
ADDCMD=${4:-""}   # must match the ADDCMD used for the OCT-only/IR-only finetune runs being combined

# "Easy" dual eval for the OCTCube-IR ablation set, over the systask dataset
# list (same list as UFcohort_multirun_systask_OCTCubeIR.sh). For each dataset,
# this locates the checkpoint-best.pth produced by:
#   - finetune_UFcohort_IRB2024v5_OCT_OCTCubeIR.sh (OCT-only, OCTCube-IR init)
#   - finetune_UFcohort_IRB2024v5_IR_OCTCubeIR.sh  (IR-only,  OCTCube-IR init)
# (run via UFcohort_multirun_systask_OCTCubeIR.sh pointed at each of those
# scripts) and submits finetune_UFcohort_IRB2024v5_dualeval_OCTCubeIR.sh with
# both paths. Datasets whose checkpoints aren't both present yet are skipped
# (printed, not submitted) so this can be re-run as runs finish.
#
# bash scripts/UFcohort_multirun_dualeval_systask_OCTCubeIR.sh scripts/finetune_UFcohort_IRB2024v5_dualeval_OCTCubeIR.sh AUPRC
# sbatch scripts/UFcohort_multirun_dualeval_systask_OCTCubeIR.sh scripts/finetune_UFcohort_IRB2024v5_dualeval_OCTCubeIR.sh AUC 500
data_type="IRB2024_v5"
CKPT_ROOT=/orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st

DATASETS=(PD_all_split DKD_all_split Diabetes_all_split)
CLASSES=(2 2 2)  # Number of classes for each dataset
TASK_MODES=(binary_cls binary_cls binary_cls)  # Task mode, can be changed as needed

for i in "${!DATASETS[@]}"
do
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    TASK_MODE="${TASK_MODES[$i]}"

    OCT_CKPT=${CKPT_ROOT}/UFcohort_${DATASET}_${data_type}_3D_st_flash_attn_nodrop_OCTCubeIR_subtr${SUBSETNUM}_${TASK_MODE}${ADDCMD}/checkpoint-best.pth
    FUNDUS_CKPT=${CKPT_ROOT}/UFcohort_${DATASET}_${data_type}_2D_flash_attn_OCTCubeIR_subtr${SUBSETNUM}_${TASK_MODE}${ADDCMD}/checkpoint-best.pth

    if [[ ! -f "$OCT_CKPT" ]] || [[ ! -f "$FUNDUS_CKPT" ]]; then
        echo "=== Skipping $DATASET: checkpoint(s) not found yet ==="
        [[ -f "$OCT_CKPT" ]]    || echo "  missing OCT checkpoint:    $OCT_CKPT"
        [[ -f "$FUNDUS_CKPT" ]] || echo "  missing FUNDUS checkpoint: $FUNDUS_CKPT"
        continue
    fi

    echo "sbatch $SCRIPT $DATASET $OCT_CKPT $FUNDUS_CKPT $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD --wandb_tags late_fusion"
    sbatch $SCRIPT $DATASET $OCT_CKPT $FUNDUS_CKPT $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD --wandb_tags late_fusion
done
