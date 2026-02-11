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
num_frames=15
TASK=finetune_oimhs_${ADDCMD}_3D_10folds_correct_${num_frames}
OUTPUT_DIR=/orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st/${TASK}/
python main_finetune_downstream_oimhs.py --nb_classes 3 \
    --data_path $ROOT/OCTCubeM/assets/ext_oph_datasets/OIMHS_dataset/cls_images/ \
    --rank -1 \
    --dataset_mode frame \
    --iterate_mode visit \
    --name_split_char _ \
    --patient_idx_loc 3 \
    --max_frames ${num_frames} \
    --num_frames ${num_frames} \
    --task ${TASK} \
    --task_mode multi_cls \
    --val_metric AUPRC \
    --k_folds 10 \
    $ADDCMD \
    --input_size 256 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --val_batch_size 8 \
    --warmup_epochs 10 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type 3D_st_flash_attn \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 150 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --finetune $ROOT/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
    --smaller_temporal_crop crop \

