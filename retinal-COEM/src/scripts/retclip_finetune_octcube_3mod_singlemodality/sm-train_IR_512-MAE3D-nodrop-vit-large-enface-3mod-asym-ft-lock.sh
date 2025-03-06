# Zixuan Zucks Liu @University of Washington
# All rights reserved
prefix=YOUR_PREFIX
echo "prefix: $prefix"
# end the shell script if any command returns a non-zero status

LAST_EPOCH=50
EPOCH=50

LOAD_OUTPUT=$HOME/$prefix/retclip_exp/

OUTPUT=$HOME/$prefix/retclip_exp_new_octcube_3mod/

single_modality=image
# single_modality=text1
# single_modality=text2
data_path=$HOME/$prefix/Ophthal/
split_path=$HOME/$prefix/Oph_cls_task/scr_train_val_test_split_622/
patient_id_list_dir=multi_label_expr_all/
resume_pt_path=${LOAD_OUTPUT}/checkpoints/epoch_${LAST_EPOCH}.pt


WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m training.main_retclip_finetune_more_cls_3mod \
    --patient_dataset \
    --cls_dataset \
    --multimodal_type oct3d_paired_faf_ir_cls \
    --data_path ${data_path} \
    --num_frames 60 \
    --split_path ${split_path} \
    --patient_id_list_dir ${patient_id_list_dir} \
    --disease POG \
    --cls_dataset_type GAGrowth \
    --task_mode multi_label  \
    --input_size 256,384 \
    --input_size_ir 384 \
    --transform_type monai_3D \
    --color_mode gray \
    --logs $OUTPUT \
    --report-to wandb \
    --workers 10 \
    --warmup 500 \
    --batch-size 2 \
    --lr 2e-5 \
    --epochs $EPOCH \
    --model vit_large_patch16_retFound_enface-vit_large_patch16_mae_joint_nodrop_3mod \
    --local-loss \
    --grad-checkpointing \
    --ddp-static-graph \
    --gather-with-grad \
    --save-frequency 10 \
    --load_non_flash_attn_to_flash_attn \
    --accum-freq 1 \
    --prefetch_factor 16 \
    --enable_3mod_training \
    --enable_independent_test \
    --not_load_epoch_when_resume \
    --single_modality ${single_modality} \
    --lock-text \
    --lock-text-unlocked-layers 9 \
    --lock-image \
    --lock-image-unlocked-groups 9 \
    # --resume ${resume_pt_path} \