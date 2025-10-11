#! /bin/bash

SCRIPT=$1
MODEL=${2:-"flash_attn_vit_large_patch16"}
Num_CLASS=${3:-"2"}
Eval_score=${4:-"AUPRC"}
SUBSETNUM=${5:-0} # 0, 500, 1000
ADDCMD=${6:-""}

#bash scripts/UFcohort_multirun_systask.sh scripts/finetune_UFcohort_IRB2024v5.sh flash_attn_vit_large_patch16 2 AUPRC
#bash scripts/UFcohort_multirun_systask.sh scripts/finetune_UFcohort_IRB2024v5.sh flash_attn_vit_large_patch16 2 AUC
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
    # Submit the job to Slurm
    sbatch $SCRIPT $DATASET $MODEL $NUM_CLASS $Eval_score $TASK_MODE $SUBSETNUM $ADDCMD
done