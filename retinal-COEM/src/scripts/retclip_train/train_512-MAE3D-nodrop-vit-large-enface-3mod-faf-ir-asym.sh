# Zixuan Zucks Liu @University of Washington
# All rights reserved

prefix=YOUR_PREFIX
echo "prefix: $prefix"
# end the shell script if any command returns a non-zero status

EPOCH=50

prefix=/


OUTPUT=$HOME/$prefix/retclip_exp/
# This two is to spcifty the path of the dataset path, the split file and the patient id list
data_path=$HOME/$prefix/Ophthal/
split_path=$HOME/$prefix/OCTCubeM/assets/Oph_cls_task/scr_train_val_test_split_622/
patient_id_list_dir=multi_label_expr_all/

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 12345 --nproc_per_node 8 -m training.main_retclip_3modalities \
    --patient_dataset \
    --combined_dataset \
    --multimodal_type oct_faf_ir \
    --data_path ${data_path} \
    --num_frames 60 \
    --split_path ${split_path} \
    --patient_id_list_dir ${patient_id_list_dir} \
    --disease POG \
    --task_mode multi_label  \
    --input_size 256,384 \
    --input_size_ir 384 \
    --transform_type monai_3D \
    --color_mode gray \
    --logs $OUTPUT \
    --report-to wandb \
    --workers 10 \
    --warmup 200 \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs $EPOCH \
    --model vit_large_patch16_retFound_enface-vit_large_patch16_mae_joint_nodrop_3mod \
    --local-loss \
    --grad-checkpointing \
    --ddp-static-graph \
    --gather-with-grad \
    --save-frequency 10 \
    --load_non_flash_attn_to_flash_attn \
    --lock-image \
    --lock-image-unlocked-groups 9 \
    --accum-freq 8 \
    --prefetch_factor 16 \
    --save_last_5 \
    --lampa_only \
    --enable_3mod_training \
