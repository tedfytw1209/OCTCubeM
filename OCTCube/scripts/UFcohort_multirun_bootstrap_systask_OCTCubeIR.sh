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

SCRIPT=${1:-"scripts/UFcohort_multirun_bootstrap_OCTCubeIR.sh"}  # per-dataset 10-seed bootstrap driver
MODEL=${2:-"flash_attn_vit_large_patch16"}
Num_CLASS=${3:-"2"}
Eval_score=${4:-"AUPRC"}
SUBSETNUM=${5:-500} # 0, 500, 1000
ADDCMD=${6:-""}

# OCTCube-IR dual-modality bootstrap over the systask studies. For each dataset,
# one 10-seed bootstrap job (seeds 1..10) is submitted via SCRIPT.
#bash scripts/UFcohort_multirun_bootstrap_systask_OCTCubeIR.sh scripts/UFcohort_multirun_bootstrap_OCTCubeIR.sh flash_attn_vit_large_patch16 2 AUC 500
DATASETS=(PD_all_split DKD_all_split Diabetes_all_split)
CLASSES=(2 2 2)  # Number of classes for each dataset
TASK_MODES=(binary_cls binary_cls binary_cls)  # Task mode, can be changed as needed
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    TASK_MODE="${TASK_MODES[$i]}"
    echo "sbatch $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD"
    # Submit the per-dataset bootstrap (10 seeds) to Slurm
    sbatch $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD
done
