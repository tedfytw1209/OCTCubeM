# Few-shot, load retfound weights
prefix=YOUR_PREFIX
LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=./outputs_ft/finetune_hcms_2DCenter_fewshot_10folds_correct_20percent/
CUDA_VISIBLE_DEVICES=3 python main_finetune_downstream_hcms_20percent.py --nb_classes 2 \
    --data_path $HOME/$prefix/OCTCubeM/assets/ext_oph_datasets/HCMS/image_resized/ \
    --dataset_mode frame \
    --iterate_mode patient \
    --name_split_char _ \
    --patient_idx_loc 6 \
    --cls_unique \
    --task ${OUTPUT_DIR} \
    --task_mode binary_cls \
    --val_metric AUPRC \
    --few_shot \
    --k_folds 10 \
    --input_size 224 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --warmup_epochs 10 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type Center2D_flash_attn \
    --epochs 150 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --finetune $HOME/$prefix/OCTCubeM/ckpt/RETFound_oct_weights.pth \
    --return_bal_acc \
    --not_dataset_random_reshuffle_patient \
    --load_non_flash_attn_to_flash_attn \
