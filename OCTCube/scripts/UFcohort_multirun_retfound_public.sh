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

ADDCMD=${1:-""}

# RETFound-center (2D single central B-scan)
bash scripts/cross-cohort/RETFound-center/finetune_duke14.sh $ADDCMD
bash scripts/cross-cohort/RETFound-center/finetune_oimhs.sh $ADDCMD
bash scripts/cross-cohort/RETFound-center/finetune_umn.sh $ADDCMD

# RETFound-all (3D with all B-scans)
bash scripts/cross-cohort/RETFound-all/finetune_duke14_effective_fold.sh $ADDCMD
bash scripts/cross-cohort/RETFound-all/finetune_oimhs.sh $ADDCMD
bash scripts/cross-cohort/RETFound-all/finetune_umn.sh $ADDCMD

# cross-device
bash scripts/cross-device/RETFound-center/finetune_glaucoma_cirrus.sh $ADDCMD
bash scripts/cross-device/RETFound-all/finetune_glaucoma_cirrus.sh $ADDCMD