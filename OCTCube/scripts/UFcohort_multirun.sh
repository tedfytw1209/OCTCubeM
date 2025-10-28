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

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_all_split_binary 2, Glaucoma_all_split_binary 2
MODEL=${2:-"flash_attn_vit_large_patch16"}
Num_CLASS=${3:-"2"}
Eval_score=${4:-"AUPRC"}
SUBSETNUM=${5:-0} # 0, 500, 1000
ADDCMD=${6:-""}

#bash scripts/UFcohort_multirun.sh scripts/finetune_UFcohort_IRB2024v5.sh flash_attn_vit_large_patch16 2 AUPRC
#sbatch scripts/UFcohort_multirun.sh scripts/finetune_UFcohort_IRB2024v5.sh flash_attn_vit_large_patch16 2 AUC 500
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split)  # List of datasets
CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
TASK_MODES=(binary_cls binary_cls multi_cls multi_cls binary_cls binary_cls)  # Task mode, can be changed as needed
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    TASK_MODE="${TASK_MODES[$i]}"
    echo "bash $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD"
    # Submit the job to Slurm
    bash $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD
done