#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module purge
module load conda
conda activate octcube

# Force Conda C++ runtime before importing PyTorch/SciPy extensions
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "LD_PRELOAD=$LD_PRELOAD"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python main_finetune_downstream_oimhs.py --nb_classes 3 --data_path /blue/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/OIMHS_dataset/cls_images/ --dataset_mode frame --iterate_mode visit --name_split_char _ --patient_idx_loc 3 --max_frames 24 --num_frames 24 --task finetune_oimhs__3D_RETFound_10folds_correct_ --task_mode multi_cls --val_metric AUPRC --k_folds 10 --input_size 224 --log_dir /blue/ruogu.fang/tienyuchang/log_pt/ --output_dir /orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st/finetune_oimhs__3D_RETFound_10folds_correct_/ --batch_size 1 --val_batch_size 8 --warmup_epochs 10 --world_size 1 --model flash_attn_vit_large_patch16_3DSliceHead --patient_dataset_type 3D_flash_attn --transform_type frame_2D --epochs 150 --blr 5e-3 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.2 --finetune /blue/ruogu.fang/tienyuchang/OCTCubeM/ckpt/RETFound_oct_weights.pth --return_bal_acc --smaller_temporal_crop crop --load_non_flash_attn_to_flash_attn