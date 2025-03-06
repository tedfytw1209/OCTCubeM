seed=1
gpu_id=0
w1=0.2
lr=1e-2
slivit_head_depth=5
LOG_DIR=$HOME/log_pt/
prefix=YOUR_PREFIX
prefix=""
slivit_out_dir=./slivit_out_auxi_slivit/slivit_out_auxi_${seed}_w1_${w1}_slivit_head_${slivit_head_depth}/
img_suffix=".npy"
path_col="path_npy"



CUDA_VISIBLE_DEVICES=${gpu_id} python -m main_finetune_downstream_inhouse_singlefold_diffmodal --nb_classes 3 \
    --single_fold \
    --num_frames 60 \
    --task_mode regression  \
    --enable_early_stop \
    --early_stop_patience 12 \
    --val_metric AUPRC \
    --input_size 256 \
    --log_dir ${LOG_DIR} \
    --batch_size 2 \
    --warmup_epochs 1 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type 3D_st_flash_attn_slivit \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 20 \
    --blr ${lr} \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --always_test \
    --finetune $HOME/$prefix/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
    --slivit_exp \
    --slivit_meta $HOME/$prefix/OCTCubeM/OCTCube/assets/us3d_meta/echonet.csv \
    --slivit_dataset us3d \
    --slivit_label EF \
    --slivit_us_auxi_reg \
    --regression_loss_name l1loss \
    --slivit_path_col $path_col \
    --slivit_img_suffix $img_suffix \
    --slivit_3_channels \
    --slivit_out_dir $slivit_out_dir \
    --seed $seed \
    --slivit_w1 $w1 \
    --slivit_vit_depth_num ${slivit_head_depth} \
