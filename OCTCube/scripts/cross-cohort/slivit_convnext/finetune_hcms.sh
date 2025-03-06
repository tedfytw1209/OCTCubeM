# Few shot 10 folds, use k frames
prefix=YOUR_PREFIX
LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=./outputs_ft_slivit_ext/finetune_hcms_3D_fewshot_10folds_correct_${num_frames}_20percent/
num_frames=18
CUDA_VISIBLE_DEVICES=3 python main_finetune_downstream_hcms_20percent.py --nb_classes 2 \
    --data_path $HOME/$prefix/OCTCubeM/assets/ext_oph_datasets/HCMS/image_resized/ \
    --dataset_mode frame \
    --iterate_mode patient \
    --name_split_char _ \
    --patient_idx_loc 6 \
    --cls_unique \
    --max_frames ${num_frames} \
    --num_frames ${num_frames} \
    --task ${OUTPUT_DIR} \
    --task_mode binary_cls \
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
    --finetune whatever \
    --model whatever \
    --patient_dataset_type convnext_slivit \
    --transform_type frame_2D \
    --color_mode rgb \
    --epochs 100 \
    --blr 1e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --always_test \
    --return_bal_acc \
    --smaller_temporal_crop crop \
    --not_dataset_random_reshuffle_patient \
