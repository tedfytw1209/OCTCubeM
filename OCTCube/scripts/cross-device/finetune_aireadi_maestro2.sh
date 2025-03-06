prefix=YOUR_PREFIX
LOG_DIR=$HOME/log_pt/
AIREADI_DIR=$HOME/$prefix/AI-READI/ # Or AI-READI-2.0
OUTPUT_DIR=./outputs_ft_st/finetune_aireadi_3D_maes_nodrop_10folds/
CUDA_VISIBLE_DEVICES=0 python -m main_finetune_downstream_aireadi_correct_visit --nb_classes 2 \
    --data_path ${AIREADI_DIR} \
    --dataset_mode dicom_aireadi \
    --iterate_mode visit \
    --num_frames 60 \
    --k_fold \
    --k_folds 10 \
    --task ${OUTPUT_DIR} \
    --task_mode binary_cls \
    --val_metric BalAcc \
    --input_size 256 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --val_batch_size 4 \
    --warmup_epochs 2 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type 3D_st_flash_attn_nodrop \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 20 \
    --blr 5e-3 \
    --layer_decay 0.7 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --num_workers 20 \
    --finetune $HOME/$prefix/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
    --always_test \
    --aireadi_location Macula \
    --aireadi_device Maestro2 \
    --aireadi_pre_patient_cohort All_have \
    --shift_mean_std \