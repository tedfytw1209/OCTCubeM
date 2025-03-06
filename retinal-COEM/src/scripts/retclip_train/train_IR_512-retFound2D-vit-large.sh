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
split_path=$HOME/$prefix/Oph_cls_task/scr_train_val_test_split_622/
patient_id_list_dir=multi_label_expr_all/


# python -m training.main_retclip \
torchrun --master_port 12345 --nproc_per_node 4 -m training.main_retclip \
    --patient_dataset \
    --dataset_type Center2D \
    --data_path $HOME/$prefix/Ophthal/ \
    --num_frames 48 \
    --split_path $split_path \
    --patient_id_list_dir $patient_id_list_dir \
    --disease POG \
    --task_mode multi_label  \
    --input_size 224 \
    --input_size_ir 224 \
    --transform_type frame_2D \
    --color_mode rgb \
    --logs $OUTPUT \
    --report-to wandb \
    --workers 10 \
    --warmup 200 \
    --batch-size 128 \
    --lr 1e-4 \
    --epochs $EPOCH \
    --model vit_large_patch16_retFound-vit_large_patch16_retFound2D \
    --local-loss \
    --grad-checkpointing \
    --ddp-static-graph \
    --gather-with-grad \
    --save-frequency 5 \
    --load_non_flash_attn_to_flash_attn \
    --lock-image \
    --lock-image-unlocked-groups 9 \
    --accum-freq 1 \
