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

# cross-device
bash scripts/cross-device/RETFound-center/finetune_glaucoma_cirrus.sh $ADDCMD
bash scripts/cross-device/RETFound-all/finetune_glaucoma_cirrus.sh $ADDCMD