#! /bin/bash

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_all_split_binary 2, Glaucoma_all_split_binary 2
MODEL=${2:-"flash_attn_vit_large_patch16"}
Num_CLASS=${3:-"2"}
Eval_score=${4:-"AUPRC"}
ADDCMD=${5:-""}

#bash scripts/UFcohort_multirun.sh scripts/finetune_UFcohort_IRB2024v4.sh flash_attn_vit_large_patch16 2 AUPRC
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split)  # List of datasets
CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
TASK_MODES=(binary_cls binary_cls multi_cls multi_cls binary_cls binary_cls)  # Task mode, can be changed as needed
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    TASK_MODE="${TASK_MODES[$i]}"
    echo "sbatch $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $ADDCMD"
    # Submit the job to Slurm
    #sbatch $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $ADDCMD
done