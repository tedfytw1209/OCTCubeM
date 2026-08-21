#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
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

# Linear-probe variant of finetune_public2D_OCTCubeIR_fundus.sh: freezes
# OCTCube-IR's pretrained 2D en-face tower (--freeze_extractor) and only
# trains the classification head, instead of fully fine-tuning the tower.
# See ../../main_finetune_downstream_public2D_OCTCubeIR_fundus.py.

# Root containing pre-split train/val/test/Class_x/ folders for these public
#   fundus datasets, per OphFoundation's reference benchmark script
#   (dataset_root/dataset_name/{train,val,test}/Class_x/).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/"

# OCTCube-IR's jointly-pretrained checkpoint; only its en-face ("text.") tower
# is used here (see load_octcubeir_2d_tower_checkpoint in the python script).
ROOT=/blue/ruogu.fang
prefix=tienyuchang
OCTCUBEIR_CKPT=${ROOT}/${prefix}/OCTCubeM/ckpt/mm_octcube_ir.pt

LOG_DIR=./log_pt/
# Separate output dir from both the test-peeking-era runs and the full
# fine-tune (val-selected) runs -- --freeze_extractor also changes the
# output-dir checksum, but keeping the base dir distinct too makes the
# frozen-extractor runs easy to find.
OUTPUT_DIR=$HOME/OCTCubeM_results/outputs_ft_public2D_octcubeir_fundus_frozen_extractor_val_selected/

# 7 datasets (name:num_class, per OphFoundation's reference benchmark params;
#   num_classes is auto-inferred by the python script from the folder
#   structure, not passed here -- listed for reference only).
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
DATASETS=(Glaucoma_fundus IDRiD_data JSIEC MESSIDOR2 PAPILA Retina APTOS2019)

# $1: DATASET
launch() {
    local DATASET=$1
    python main_finetune_downstream_public2D_OCTCubeIR_fundus.py \
        --data_set ${DATASET} \
        --data_root ${DATA_ROOT} \
        --finetune ${OCTCUBEIR_CKPT} \
        --log_dir ${LOG_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --freeze_extractor \
        --save_model \
        --val_metric AUPRC \
        --return_bal_acc \
        --batch_size 32 \
        --val_batch_size 32 \
        --epochs 100 \
        --warmup_epochs 10 \
        --blr 5e-3 \
        --layer_decay 0.65 \
        --weight_decay 0.05 \
        --drop_path 0.2 \
        --input_size 224 \
        --world_size 1 \
        --overwrite \
        --rank -1
}

for DATASET in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DATASET} ==="
    launch "${DATASET}"
    echo "=== Dataset ${DATASET} done ==="
done
