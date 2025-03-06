# Few shot 10 folds, use k frames
prefix=YOUR_PREFIX
LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=./outputs_ft/finetune_umn_3D_fewshot_10folds_correct/
num_frames=24
CUDA_VISIBLE_DEVICES=3 python main_finetune_downstream_umn.py --nb_classes 2 \
    --data_path $HOME/$prefix/OCTCubeM/assets/ext_oph_datasets/UMN/UMN_dataset/image_classification/ \
    --dataset_mode frame \
    --iterate_mode patient \
    --name_split_char _ \
    --patient_idx_loc 2 \
    --cls_unique \
    --max_frames ${num_frames} \
    --num_frames ${num_frames} \
    --task ${OUTPUT_DIR} \
    --task_mode binary_cls \
    --val_metric AUPRC \
    --few_shot \
    --k_folds 10 \
    --input_size 224 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --val_batch_size 8 \
    --warmup_epochs 10 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16_3DSliceHead \
    --patient_dataset_type 3D_flash_attn \
    --transform_type frame_2D \
    --epochs 150 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --finetune $HOME/$prefix/OCTCubeM/ckpt/RETFound_oct_weights.pth \
    --return_bal_acc \
    --smaller_temporal_crop crop \
    --load_non_flash_attn_to_flash_attn \
