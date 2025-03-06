seed=1
gpu_id=0
LOG_DIR=$HOME/log_pt/
prefix=YOUR_PREFIX
slivit_out_dir=./slivit_out_retfound_${seed}/
CUDA_VISIBLE_DEVICES=${gpu_id} python -m main_finetune_downstream_inhouse_singlefold_diffmodal --nb_classes 2 \
    --single_fold \
    --num_frames 60 \
    --task_mode binary_cls  \
    --enable_early_stop \
    --early_stop_patience 8 \
    --val_metric AUPRC \
    --input_size 256 \
    --log_dir ${LOG_DIR} \
    --batch_size 1 \
    --warmup_epochs 1 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16_3DSliceHead \
    --patient_dataset_type 3D_flash_attn \
    --transform_type monai_3D \
    --load_non_flash_attn_to_flash_attn \
    --color_mode gray \
    --epochs 50 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --always_test \
    --finetune $HOME/$prefix/OCTCubeM/ckpt/RETFound_oct_weights.pth \
    --return_bal_acc \
    --slivit_exp \
    --slivit_meta $HOME/$prefix/OCTCubeM/OCTCube/assets/us3d_meta/echonet.csv \
    --slivit_medmnist_root $HOME/$prefix/OCTCubeM/OCTCube/assets/medmnist_data/ \
    --slivit_dataset ct3d \
    --slivit_out_dir $slivit_out_dir \
    --slivit_slices 64 \
    --seed $seed \
    --slivit_3_channels \
