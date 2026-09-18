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

module load conda
conda activate octcube

STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_all_split_binary 2, Glaucoma_all_split_binary 2
MODEL=${2:-"flash_attn_vit_large_patch16"}
Num_CLASS=${3:-"2"}
Eval_score=${4:-"AUC"}
TASK_MODE=${5:-"binary_cls"}
SUBSETNUM=${6:-0} # 0, 500, 1000
ADDCMD=${7:-""}

# OCT-only ablation of the OCTCube-IR dual-modality experiment: only the 3D OCT
# tower is fine-tuned, initialized from OCTCube-IR's jointly-pretrained
# checkpoint (mm_octcube_ir.pt) instead of the single-modality OCTCube.pth
# (cf. finetune_UFcohort_IRB2024v5.sh). main_finetune_downstream_UFcohort.py
# extracts the 'visual.*' (OCT tower) slice of the checkpoint when
# --patient_dataset_type is a 3D_st* type and the checkpoint has a 'state_dict'
# key (see the OCTCube-IR branch added for this in that script).
# The companion IR-only run is finetune_UFcohort_IRB2024v5_IR_OCTCubeIR.sh;
# together their checkpoint-best.pth outputs feed
# finetune_UFcohort_IRB2024v5_dualeval_OCTCubeIR.sh for the "easy" dual eval.
#
# Example usage:
#   sbatch scripts/finetune_UFcohort_IRB2024v5_OCT_OCTCubeIR.sh AMD_all_split flash_attn_vit_large_patch16 2 AUC binary_cls 0
data_type="IRB2024_v5"
dataset_type="3D_st_flash_attn_nodrop"
model_tag="OCTCubeIR"
ROOT=/blue/ruogu.fang
prefix=tienyuchang
IMG_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/
CSV_DIR=/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv
LOG_DIR=./log_pt/
TASK=UFcohort_${STUDY}_${data_type}_${dataset_type}_${model_tag}_subtr${SUBSETNUM}_${TASK_MODE}${ADDCMD}/
OUTPUT_DIR=/orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st/${TASK}
MM_CKPT=${ROOT}/${prefix}/OCTCubeM/ckpt/mm_octcube_ir.pt

python main_finetune_downstream_UFcohort.py --nb_classes $Num_CLASS \
    --data_path $IMG_DIR \
    --csv_path $CSV_DIR \
    --rank -1 \
    --dataset_mode frame \
    --iterate_mode visit \
    --name_split_char - \
    --patient_idx_loc 1 \
    --max_frames 25 \
    --num_frames 60 \
    --few_shot \
    --k_folds 0 \
    --task ${TASK} \
    --task_mode $TASK_MODE \
    --val_metric $Eval_score \
    --input_size 128 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --val_batch_size 16 \
    --warmup_epochs 10 \
    --world_size 1 \
    --model $MODEL \
    --patient_dataset UFcohort \
    --patient_dataset_type $dataset_type \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 100 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --num_workers 16 \
    --finetune ${MM_CKPT} \
    --new_subset_num $SUBSETNUM \
    --return_bal_acc \
    --not_print_logits \
    --save_model \
    ${ADDCMD}
