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

module load conda
conda activate octcube

# 10 folds, use k frames
ADDCMD=${1:-""}

ROOT=/blue/ruogu.fang/tienyuchang
LOG_DIR=$ROOT/log_pt/
TASK=finetune_oimhs_${ADDCMD}_2DCenter_10folds_correct
OUTPUT_DIR=/orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st/${TASK}/

python main_finetune_downstream_oimhs.py --nb_classes 3 \
    --data_path $ROOT/OCTCubeM/assets/ext_oph_datasets/OIMHS_dataset/cls_images/ \
    --rank -1 \
    --dataset_mode frame \
    --iterate_mode visit \
    --name_split_char _ \
    --patient_idx_loc 3 \
    --task ${TASK} \
    --task_mode multi_cls \
    --val_metric AUPRC \
    $ADDCMD \
    --k_folds 10 \
    --input_size 224 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --val_batch_size 8 \
    --warmup_epochs 10 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type Center2D_flash_attn \
    --transform_type frame_2D \
    --epochs 150 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --finetune $ROOT/OCTCubeM/ckpt/RETFound_oct_weights.pth \
    --return_bal_acc \
    --load_non_flash_attn_to_flash_attn \
