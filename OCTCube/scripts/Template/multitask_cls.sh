prefix=YOUR_PREFIX
disease=AMD
k_folds=5
LOG_DIR=$HOME/log_pt/
OUTPUT_DIR=./outputs_ft_st/finetune_inhouse_multi_task_default_3D_correct_patient_singlefold_balAcc_model/
CUDA_VISIBLE_DEVICES=2 python -m main_finetune_downstream_inhouse_singlefold --nb_classes 16 \
    --data_path $HOME/$prefix/Ophthal/ \
    --task ${OUTPUT_DIR} \
    --single_fold \
    --k_folds $k_folds \
    --num_frames 48 \
    --split_path $HOME/$prefix/OCTCubeM/assets/Oph_cls_task/scr_train_val_test_split_622/ \
    --patient_id_list_dir $HOME/$prefix/OCTCubeM/assets/multi_cls_expr_10x/ \
    --disease ${disease} \
    --task_mode multi_task_default  \
    --enable_early_stop \
    --early_stop_patience 10 \
    --val_metric AUPRC \
    --input_size 256 \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --warmup_epochs 2 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type 3D_st_flash_attn \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 10 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --always_test \
    --finetune $HOME/$prefix/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
    # --save_model \
    # --resume latest \
