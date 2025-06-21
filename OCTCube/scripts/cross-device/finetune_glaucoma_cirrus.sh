ROOT=/blue/ruogu.fang
prefix=tienyuchang
LOG_DIR=$ROOT/log_pt/
OUTPUT_DIR=./outputs_ft_st/finetune_glaucoma_3D_fewshot_10folds_correct_visit/
CUDA_VISIBLE_DEVICES=1 python main_finetune_downstream_glaucoma_correct_visit.py --nb_classes 2 \
    --data_path $ROOT/$prefix/OCTCubeM/assets/ext_oph_datasets/glaucoma_processed/ \
    --dataset_mode volume \
    --iterate_mode visit \
    --name_split_char - \
    --patient_idx_loc 1 \
    --max_frames 60 \
    --num_frames 60 \
    --few_shot \
    --k_folds 10 \
    --task ${OUTPUT_DIR} \
    --task_mode binary_cls \
    --val_metric AUPRC \
    --input_size 128 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --val_batch_size 16 \
    --warmup_epochs 5 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type 3D_st_flash_attn_nodrop \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 100 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --num_workers 10 \
    --finetune $ROOT/$prefix/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
