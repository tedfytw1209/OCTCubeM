#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module purge
module load conda
conda activate octcube

SCRIPT="scripts/finetune_UFcohort_IRB2024v5_bootstrap.sh"
DATASET=$1 #DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split DR_filtered_all_split DR_fbinary_all_split Glaucoma_filtered_all_split Glaucoma_fbinary_all_split) 
MODEL=${2:-"flash_attn_vit_large_patch16"}
NUM_CLASS=${3:-"2"}
Eval_score=${4:-"AUPRC"}
TASK_MODE=${5:-"binary_cls"}
SUBSETNUM=${6:-500} # 0, 500, 1000
ADDCMD=${7:-""}

#sbatch scripts/UFcohort_multirun_bootstrap_tmp.sh AMD_all_split flash_attn_vit_large_patch16 2 AUC binary_cls 500
SUBSET_SEEDS=(8 9 10)
for i in "${!SUBSET_SEEDS[@]}"
do
    # Create a job name based on the variables
    SUBSETSEED="${SUBSET_SEEDS[$i]}"
    echo "bash $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $SUBSETSEED $ADDCMD"
    # Submit the job to Slurm
    bash $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $SUBSETSEED $ADDCMD
done