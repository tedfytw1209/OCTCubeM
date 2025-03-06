# 24 frames, Few-shot
prefix=YOUR_PREFIX
prefix=""
LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=./outputs_ft_slivit_ext/finetune_duke14_3D_fewshot_10folds_effective_fold_fixrandom/
CUDA_VISIBLE_DEVICES=3 python main_finetune_downstream_duke14.py --nb_classes 3 \
    --data_path $HOME/$prefix/OCTCubeM/assets/ext_oph_datasets/DUKE_14_Srin/duke14_processed/ \
    --dataset_mode frame \
    --iterate_mode patient \
    --name_split_char _ \
    --patient_idx_loc 1 \
    --cls_unique \
    --max_frames 24 \
    --num_frames 24 \
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
    --not_dataset_random_reshuffle_patient \
