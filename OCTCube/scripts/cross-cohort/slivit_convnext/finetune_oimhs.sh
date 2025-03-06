# 15 frames, Few-shot
prefix=YOUR_PREFIX
LOG_DIR=$HOME/log_pt/
num_frames=15
OUTPUT_DIR=./outputs_ft_slivit_ext/finetune_oimhs_3D_fewshot_10folds_correct_${num_frames}/
CUDA_VISIBLE_DEVICES=3 python main_finetune_downstream_oimhs.py --nb_classes 3 \
    --data_path $HOME/$prefix/OCTCubeM/assets/ext_oph_datasets/OIMHS_dataset/cls_images/ \
    --dataset_mode frame \
    --iterate_mode visit \
    --name_split_char _ \
    --patient_idx_loc 3 \
    --max_frames ${num_frames} \
    --num_frames ${num_frames} \
    --few_shot \
    --k_folds 10 \
    --task ${OUTPUT_DIR} \
    --task_mode multi_cls \
    --val_metric AUPRC \
    --input_size 256 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --val_batch_size 8 \
    --warmup_epochs 10 \
    --world_size 1 \
    --finetune whatever \
    --model whatever \
    --patient_dataset_type convnext_slivit \
    --transform_type frame_2D \
    --color_mode rgb \
    --epochs 150 \
    --blr 1e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --return_bal_acc \
    --smaller_temporal_crop crop \
