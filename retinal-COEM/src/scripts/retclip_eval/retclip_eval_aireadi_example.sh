# Zixuan Zucks Liu @University of Washington
# All rights reserved
prefix=YOUR_PREFIX
echo "prefix: $prefix"

OUTPUT=YOUR_PATH
mm_ckpt_path=OCTCubeM/ckpt/mm_octcube_ir.pth



data_path=$HOME/$prefix/AI-READI/
split_path=$HOME/$prefix/Oph_cls_task/scr_train_val_test_split_622/
patient_id_list_dir=multi_label_expr_all/
CUDA_VISIBLE_DEVICES=0 python -m training.main_retclip \
    --patient_dataset \
    --data_path ${data_path} \
    --num_frames 54 \
    --split_path ${split_path} \
    --patient_id_list_dir ${patient_id_list_dir} \
    --disease POG \
    --task_mode multi_label  \
    --input_size 256 \
    --input_size_ir 224 \
    --transform_type monai_3D \
    --color_mode gray \
    --logs $OUTPUT \
    --report-to wandb \
    --workers 10 \
    --warmup 200 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs $EPOCH \
    --model vit_large_patch16_retFound-vit_large_patch16_mae_joint_nodrop \
    --local-loss \
    --grad-checkpointing \
    --ddp-static-graph \
    --gather-with-grad \
    --save-frequency 10 \
    --load_non_flash_attn_to_flash_attn \
    --lock-image \
    --lock-image-unlocked-groups 9 \
    --accum-freq 4 \
    --prefetch_factor 8 \
    --evaluate_only \
    --resume ${OUTPUT}/$mm_ckpt_path \
    --return_metainfo \
    --return_patient_id \
    --save_retrieval_results \
    --evaluate_all \
    --aireadi_location Macula \
    --aireadi_device Spectralis \
    --aireadi_pre_patient_cohort All \
    --aireadi_only_include_pair \
    --dataset_mode dicom_aireadi \