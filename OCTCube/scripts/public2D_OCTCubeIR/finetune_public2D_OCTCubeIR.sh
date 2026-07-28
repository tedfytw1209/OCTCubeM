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

# Fine-tune + evaluate OCTCube-IR's 2D en-face tower (see
# ../../main_finetune_downstream_public2D_OCTCubeIR.py) on all four public
# single-B-scan classification datasets that already have a Center2D_flash_attn
# pipeline in this repo. Loops over datasets one at a time (like MIRAGE's
# run_fundus_all_tasks_l4.sh), running the full 10-fold CV per dataset.
#
# Root containing the per-dataset processed folders, one level above each
# dataset's subdir baked into DATASET_CONFIGS in the python script
# (DUKE_14_Srin/duke14_processed/, OIMHS_dataset/cls_images/, etc.) -- see
# assets/BENCHMARK.md for how to obtain/process each dataset.
DATA_ROOT=$HOME/OCTCubeM/assets/ext_oph_datasets/

# OCTCube-IR's jointly-pretrained checkpoint; only its en-face ("text.") tower
# is used here (see load_octcubeir_2d_tower_checkpoint in the python script).
OCTCUBEIR_CKPT=$HOME/OCTCubeM/ckpt/mm_octcube_ir.pt

LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=$HOME/OCTCubeM_results/outputs_ft_public2D_octcubeir/

DATASETS=(duke14 oimhs umn glaucoma)

# $1: DATASET
launch() {
    local DATASET=$1
    local BATCH_SIZE VAL_BATCH_SIZE EPOCHS WARMUP_EPOCHS

    # Per-dataset batch size / schedule, taken from the existing
    # scripts/cross-{cohort,device}/RETFound-center/*.sh launchers.
    case $DATASET in
        duke14)   BATCH_SIZE=4; VAL_BATCH_SIZE=8;  EPOCHS=150; WARMUP_EPOCHS=10 ;;
        oimhs)    BATCH_SIZE=8; VAL_BATCH_SIZE=8;  EPOCHS=150; WARMUP_EPOCHS=10 ;;
        umn)      BATCH_SIZE=2; VAL_BATCH_SIZE=8;  EPOCHS=150; WARMUP_EPOCHS=10 ;;
        glaucoma) BATCH_SIZE=4; VAL_BATCH_SIZE=16; EPOCHS=100; WARMUP_EPOCHS=5  ;;
    esac

    python main_finetune_downstream_public2D_OCTCubeIR.py \
        --data_set ${DATASET} \
        --data_root ${DATA_ROOT} \
        --finetune ${OCTCUBEIR_CKPT} \
        --log_dir ${LOG_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --k_folds 10 \
        --val_metric AUPRC \
        --return_bal_acc \
        --batch_size ${BATCH_SIZE} \
        --val_batch_size ${VAL_BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --blr 5e-3 \
        --layer_decay 0.65 \
        --weight_decay 0.05 \
        --drop_path 0.2 \
        --input_size 224 \
        --world_size 1 \
        --rank -1
}

for DATASET in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DATASET} ==="
    launch "${DATASET}"
    echo "=== Dataset ${DATASET} done ==="
done
