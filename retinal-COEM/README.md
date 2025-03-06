# OCTCube-M multi modal training and evlauation 
This repo contains the code for specifically, the OCTCube-IR and OCTCube-EF training and evluation.

## Checklist:
- [] Add and test AI-READI dataset for OCTCube-EF multi modal training and evaluation.
- [x] Release the model for OCTCube-IR model for AI-READI dataset.
- [x] Example AI-READI dataset guidance to improve code-starting experience.
- [x] Dataset.md and README.md for better understanding of the dataset and the code (we are still refining it, welcome to any suggestions).
- [x] Upload code for OCTCube-IR and OCTCube-EF multi modal training and evaluation.


## Multi-modal training

To run the OCTCube-IR training, at `OCTCubeM/retinal-COEM/retClip/src`, run 
``` source scripts/retclip/train_IR_512-MAE3D-nodrop-vit-large.sh ```

To run the RETFound center training, at `OCTCubeM/retinal-COEM/retClip/src`, run 
``` source scripts/retclip/train_IR_512-retfound2D-vit-large.sh ```

To run the RETFouhnd all training, at `OCTCubeM/retinal-COEM/retClip/src`, run 
``` source scripts/retclip/train_IR_512-retfound3D-vit-large.sh ```

To run the OCTCube-EF training, at `OCTCubeM/retinal-COEM/src`, run 
``` source scrtips/train_512-MAE3D-nodrop-vit-large-enface-3mod-faf-ir-asym.sh ```

## Fine-tuning on multi-modal GA prognosis (or other tasks)

To fientune the OCTCube-EF on your own multi-modal GA dataset, at `OCTCubeM/retinal-COEM/src`, run 
``` source scripts/retClip_finetune_3mod/train_IR_512-MAE3D-nodrop-vit-large-enface-3mod-asym-ft.sh```

To fientune the OCTCube (only single modality is effective) on your own multi-modal, at `OCTCubeM/retinal-COEM/src`, run 
``` source scripts/retClip_finetune_3mod_singlemodality/sm-train_IR_512-MAE3D-nodrop-vit-large-enface-3mod-asym-ft-lock.sh```

**Note:** We provide the option of freezing some parameters of the model encoders, please check the configuration description below [configuration](#model-config).

To use the RETFound-all to fientune the OCTCube-EF on your own multi-modal GA dataset, at `OCTCubeM/retinal-COEM/src`, run 
``` source scripts/retClip_finetune_retfound3D_3mod/train_IR_512-retFound3D-vit-large-enface-3mod-asym-ft.sh```


## Run the evaluation

### Generate the retrieval results
To run the evaluation the OCTCube-IR model on AI-READI dataset (with your trained checkpoint), at `OCTCubeM/retinal-COEM/src`, run

``` source scripts/retClip_eval/retclip_eval_aireadi_example.sh```

You will see the retrieval results in the `./your_results/` folder. The `reesults.jsonl` should contain the retrieval results for each query image. If you want to further test on laterality prediction, remember to set `--save_retrieval_results` to `True`.

### Evaluating the metrics
To run the laterality evaluation on test set, at `OCTCubeM/retinal-COEM/src/retDisease_eval/, run

``` python evaluate_results_test_train_visualize_all_models_top3_col_aireadi_laterality```

Remember to set the correct `model_exper_name`, correct `Ophthal_dir`, `retClip_directory`, `retClip_exp_directory` and `retrieval_results_dir` in the `evaluate_results_test_train_visualize_all_models_top3_col_aireadi_laterality.py` file.


# Model config

The model will be loaded in OCTCube/Multi-modal/src/open_clip/model_configs/

The text tower (configured in text_cfg) are modified to be adopted to the enface (IR) modality. 

The image tower are modifited to be adopted with 3D OCT model.

# Training Script Parameters for OCTCube-EF

This documentation provides detailed explanations of the parameters used in the RETClip training script. The script integrates multi-modal data for the task of disease classification using a Vision Transformer model.

## Setup and Environment
- `EPOCH`: Total number of training epochs.
- `prefix`: Base path used to derive the paths for dataset and other resources.
- `YOUR_SPLIT_PATH`: Path to the file containing the dataset splits (training, validation, test).
- `YOUR_DATASET_PATH`: Root directory where the dataset is stored.
- `YOUR_PATIENT_ID_LIST_DIR`: Directory containing files with lists of patient IDs, used to fetch patient-specific data.
- `OUTPUT`: Directory where all outputs like logs and saved models will be stored.

## Training Configuration
- `--patient_dataset`: Flag to indicate the use of a patient-specific dataset.
- `--data_path`: Path to the dataset.
- `--num_frames`: Number of frames per case or scan to process.
- `--split_path`: Path to the data split file.
- `--patient_id_list_dir`: Path to the directory containing patient ID lists.
- `--disease`: Type of disease the model should focus on (e.g., POG).
- `--task_mode`: Specifies the classification mode (`multi_label` in this case).
- `--input_size`: Dimension of the input image (assumes square images).
- `--input_size_ir`: Dimension of the input for IR images, differentiating from the main input size if necessary.
- `--transform_type`: Type of transformation to apply on the data (e.g., `monai_3D`).
- `--color_mode`: Specifies the color mode of the input images (`gray` in this case).
- `--logs`: Path to save training logs.
- `--report-to`: Destination for reporting metrics and logs (e.g., weights and biases).

## Performance and Efficiency
- `--workers`: Number of worker threads for loading data.
- `--warmup`: Number of warm-up steps before actual training starts.
- `--batch-size`: Number of samples in each batch.
- `--lr`: Learning rate for the optimizer.
- `--epochs`: Total number of epochs for training.
- `--model`: Specifies the model architecture to use.
- `--local-loss`: Enables calculation of a local loss function, if applicable.
- `--grad-checkpointing`: Enables gradient checkpointing to save memory during training.
- `--ddp-static-graph`: Use static graph optimization in distributed data parallel.
- `--gather-with-grad`: Enables gathering gradients across different nodes.
- `--save-frequency`: Frequency of saving the trained model.
- `--load_non_flash_attn_to_flash_attn`: Allows loading weights from non-flash attention models to flash attention models.
- `--lock-image`: Locks the image tower during training to prevent updates.
- `--lock-image-unlocked-groups`: Number of unlocked groups in the image tower if the lock is partial.
- `--accum-freq`: Frequency of gradient accumulation steps.
- `--prefetch_factor`: Number of batches to prefetch when loading data.


# Naming of image and text and their relationships to 3D OCT and 2D *en face* fundus images
When accommodating the original open_clip framework, we deliberately maintain the original name of defining 'image' tower and 'text' tower. In COEP, we refer 'image' to be the OCT encoder, and refer 'text' to be the *en face* fundus image encoder, for both simplicity to understand the multi-modal structure, as well as maintain the flexibility to incorporate real text tower to make it a real multi-modal framework in the future. We also plan to completely split *en face* encoder out with a more proper name in the future, stay tuned!

## How can I better understand the code?
The major modifications we made to adapt open_clip to COEP are in the following files:

`retinal-COEM/src/open_clip/factory.py` - This file is the entry point for the model. It defines the model architecture and the forward pass.

`retinal-COEM/src/open_clip/model_configs/` - This directory contains the model configurations for the text and image towers. The text tower is modified to be adopted to the *en face* modality, while the image tower is modified to be adopted with 3D OCT model.

`retinal-COEM/src/open_clip/losses.py` - This file contains the loss functions used in the model. We have added our Tri-COEP loss function here (ThreeModalityClipLoss class) to calculate the local loss.

`retinal-COEM/src/open_clip/model.py` - This file contains the utlity of defining our supported ViT model (both en face ViT and OCTCube w/ flash-attn). For each model, please check the following list:
`retinal-COEM/src/open_clip/model.py` - This file contains the utility of defining our supported ViT model (both en face ViT and OCTCube w/ flash-attn). For each model, please check the following list:

-  
  `models_vit_st_flash_attn_no_drop.py` - This file contains the definition of the OCTCube model with flash attention and no dropout (default for the OCT encoder in OCTCube-M).

-  
  `models_vit_st_flash_attn.py` - This file contains the definition of the OCTCube model with flash attention.

-  
  `models_vit_3dhead_flash_attn.py` - This file contains the definition of the RETFound-all model with flash attention (default for RETFound-all OCT encoder).

-  
  `models_vit_flash_attn.py` - This file contains the definition of the RETFound-enface model with flash attention (default for the en face encoder for OCTCube-IR).

-  
  `models_vit_flash_attn_2mod.py` - This file contains the definition of the 2 modalities version of en face encoder used in OCTCube-EF model with flash attention (default for OCTCube-EF).


`retinal-COEM/src/training/data.py` - This file contains the data loading and processing functions. 

We have added the Tri-COEP data loader here to load the multi-modal data. If you want to use your own dataset, check how we add support for our dataset in this file. Specifically, check `get_patient_dataset_classification()` and `get_patient_dataset_combined()` functions.

**Note:** For more details on the dataset organization, please check the `DATASET.md` file.






## References:
#### The code is adopted and modified based on the following repositories:
#### DeiT: https://github.com/facebookresearch/deit
#### BEiT: https://github.com/microsoft/unilm/tree/master/beit
#### MAE: https://github.com/facebookresearch/mae/tree/main
#### MAE_ST: https://github.com/facebookresearch/mae_st
#### RETFOUND_MAE: https://github.com/rmaphoh/RETFound_MAE
