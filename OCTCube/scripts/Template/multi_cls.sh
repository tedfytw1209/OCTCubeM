# Few shot 10 folds, use k frames
prefix=YOUR_PREFIX
LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=./outputs_ft_st/finetune_duke14_3D_fewshot_10folds_effective_fold/
num_frames=24
CUDA_VISIBLE_DEVICES=0 python main_finetune_downstream_duke14.py --nb_classes 3 \
    --data_path $HOME/$prefix/OCTCubeM/assets/ext_oph_datasets/DUKE_14_Srin/duke14_processed/ \
    --dataset_mode frame \
    --iterate_mode patient \
    --name_split_char _ \
    --patient_idx_loc 1 \
    --cls_unique \
    --max_frames ${num_frames} \
    --num_frames ${num_frames} \
    --task ${OUTPUT_DIR} \
    --task_mode multi_cls \
    --val_metric AUPRC \
    --few_shot \
    --k_folds 10 \
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
    --finetune $HOME/$prefix/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
    --smaller_temporal_crop crop \
    --not_dataset_random_reshuffle_patient \
