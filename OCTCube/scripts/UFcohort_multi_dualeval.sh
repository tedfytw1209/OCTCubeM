#! /bin/bash

sbatch scripts/finetune_UFcohort_IRB2024v5_dualeval.sh AMD_all_split /orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st/UFcohort_AMD_all_split_IRB2024_v5_binary_cls/checkpoint-00004.pth /orange/ruogu.fang/tienyuchang/OCTCube_results/outputs_ft_st/UFcohort_AMD_all_split_IRB2024_v5_2D_flash_attn_binary_cls/checkpoint-00059.pth 2 AUPRC binary_cls

