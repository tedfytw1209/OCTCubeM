
# Fine-tune using OCTCube🚀
In this document, we provide instructions on how to fine-tune the `OCTCube` model on your own dataset. Note, for simplicity, we include several `main_finetune_downstream_*.py` to accomodate different dataset configurations. Despite some of the content in these scripts may looks overlapping, we believe these scripts are useful for you to adapt the code to your own dataset under different setting. For a more detailed desription where at high level, which script would be a better choice to accomondate for your cases, feel free to send out inquiries to us, we are happy to help you on board!


## Data preparation
### Prepare public datasets
Prepare the dataset following `OCTCubeM/assets/BENCHMARK.md`.

### Prepare in-house datasets
Prepare and organize your in-house dataset following `OCTCubeM/OCTCube/DATASET.md`.

#### NOTE: For customized dataset used for fine-tuning, check the PatentDataset.py and PatientDataset_inhouse.py to see how to load the dataset.

## Understand how to find the script you need:
We provide a bunch of useful scripts to reproduce our experiments and templates for you to easily adapt them onto your own dataset. Specifically, under `OCTCubeM/OCTCube/scripts/`:
- `cross-cohorts/` - cross-cohort evaluation for 4 baselines: `OCTCube`, `RETFound-all`, `RETFound-center`, `slivit_convnet`.
    - `finetune_duke14_*` - fine-tune on Duke14 dataset
    - `finetune_hcms_*` - fine-tune on HCMS dataset
    - `finetune_oimhs_*` - fine-tune on OIMHS dataset
    - `finetune_umn_*` - fine-tune on UMN dataset
-  `cross-device/` - cross-device evaluation for `OCTCube`.
    - `finetune_aireadi_maestro2` - fine-tune on AI-READI Maestro2 dataset
    - `finetune_glaucoma_cirrus` - fine-tune on Glaucoma Cirrus dataset
- `cross-modality/` - cross-modal evaluation for `OCTCube` and `RETFound`.
    - `finetune_ct3d` - fine-tune on CT 3D dataset
    - `finetune_us3d_b` - fine-tune on Ultrasound 2D + T dataset (classification)
    - `finetune_us3d_reg` - fine-tune on Ultrasound 2D + T dataset (regression)
- `Template/` - template for fine-tuning on your own dataset
    - `binary_cls_fewshot_10folds` - fine-tune on your own dataset with 10-fold cross-validation (where the held-out fold is used for training!)
    - `binary_cls_standard_5folds` - fine-tune on your own dataset with 5-fold cross-validation
    - `multi_cls` - fine-tune on your own dataset with multi-class classification
    - `multilabel_cls` - fine-tune on your own dataset with multi-label classification
    - `multitask_cls` - fine-tune on your own dataset with multi-task classification
    - `regression` - fine-tune on your own dataset with regression


## Run the scripts🚀

This document details the parameters used in the script for fine-tuning models specific to medical imaging, particularly focusing on OCT data and their various configurations for deep learning models.

Usually, once you have prepared your data, at `OCTCubeM/OCTCube/`, you can just run the script in `OCTCubeM/OCTCube/scripts/Template/` with the following command:

```source scripts/cross-cohort/OCTCube/finetune_duke14_effective_folds```

## Script Parameters Documentation
Below we provide a detailed description of the parameters used in the scripts for fine-tuning models.

### Common Parameters Across Scripts
- `LOG_DIR`: **Log Directory** - Directory to save logs.
- `OUTPUTDIR`: **Output Directory** - Base directory for saving outputs.
- `disease`: **Disease Type** - Specifies the disease type for which the model is being trained or fine-tuned.
- `prefix`: **Project Path Prefix** - Base path for all relative paths in the script.
- `YOUR_SPLIT_PATH`: **Data Split File Path** - Path to the file that contains data split information.

### Model and Data Handling
- `--nb_classes`: Number of classes (e.g., binary classification has 2).
- `--num_frames`: Number of frames per video or 3D scan to use.
- `--single_fold`: Indicates using a single fold for cross-validation.
- `--enable_early_stop`: Enable early stopping to prevent overfitting.
- `--early_stop_patience`: Number of epochs to wait for improvement before stopping.
- `--task_mode`: Mode of the task (`binary_cls` for binary classification). We provide `binary_cls`, `multi_cls`, `multi_label`, `multi_task`, `multi_task_default`, `regression`. Check all configuration in `main_finetune_downstream_singlefold_diffmodel.py`.
- `--val_metric`: Validation metric to monitor during training (e.g., AUPRC).
- `--warmup_epochs`: Number of warm-up epochs for learning rate scheduling.
- `--world_size`: Number of distributed world size in case of using distributed training.
- `--model`: Specifies the model architecture, used for specifying the instance pre-configured in our model construction.
- `--patient_dataset_type`: Type of patient dataset to use (e.g., `3D_st_flash_attn`). This argument is used to specify the model type we are using. For more details, please check all python scripts with the name starting with `models_vit*`.
- `--transform_type`: Type of data transformation to apply (e.g., `monai_3D` if the data volume is loaded from a single file, such as `.npy` or `.dcm`. If it is from a sequence of 2D images, use `frame_2D`).
- `--color_mode`: Color mode of the input images (`rgb` or `gray`). The default for `OCTCube` is `gray`. If you are using `RETFound` style 3-channel model, use `rgb`.
- `--layer_decay`: Layer-wise decay factor for the learning rate.
- `--weight_decay`: Weight decay factor for regularization.
- `--drop_path`: Drop path rate for regularization.
- `--return_bal_acc`: Flag to return balanced accuracy metric.
- `--smaller_temporal_crop`: To select how to crop the temporal dimension of the 3D data if you pick a small `--num_frames` and `--max_frames`. If you set it to be `crop`, then the volume is cropped to the center of the temporal dimension. If you set it to be `interp`, then the volume is interpolated to the `--num_frames` size.

### Additional Parameters for Fine-tuning
- `--finetune`: Path to the pre-trained model checkpoint for fine-tuning.
- `--output_dir`: Directory to store the fine-tuned model and outputs.
- `--log_dir`: Directory to save logs.
- `--always_test`: Always test the model after training (if applicable).
- `--load_non_flash_attn_to_flash_attn`: Specific parameter for loading weights from a non-flash attention model to a flash attention model.
- `--single_fold`: Only run a single fold of cross-validation. This is especially useful when you have a large dataset and don't want to run cross-validation for every fold.
- `--k_fold`: Enable k-fold cross-validation. Usually, this argument is exclusive with `--single_fold`.
- `--k_folds`: Number of folds for cross-validation.
- `--fewshot`: Enable few-shot learning. If enabled, then the held-out fold is used for training.
- `--downsample_normal`: Downsample normal cases in the dataset.
- `--downsample_normal_factor`: Factor by which to downsample normal cases.
- `--focal_loss`: Use focal loss for training.
- `--always_test`: Always run test after every epoch if specified.
- `--linear_probe`: Enable linear probing during training.
- `--save_model_every`: Save the model every specified number of epochs.
- `--not_print_logits`: Do not print logits during evaluation.
- `--not_save_figs`: Do not save figures during evaluation.

### More Dataset Configuration
- `--data_path`: Path to the dataset.
- `--iterate_mode`: Mode to iterate over the patient dataset (`visit` or `patient`). `visit` is usually used for longitudinal dataset or dataset with both eyes of a patient. `patient` is used if you are certain that each patient has only one instance in the dataset.
- `--dataset_mode`: Mode of the dataset (e.g., `frame` or `volume`). If your volume dataset is stored in a sequence of 2D images, use `frame` mode. Otherwise, use `volume` mode. 
- `--name_split_char`: Character used to split names in the dataset files.
- `--patient_idx_loc`: Location index of the patient ID in the filename.
- `--visit_idx_loc`: Location index of the visit ID in the filename (optional).
- `--cls_unique`: Use unique class labels for each dataset instance.
- `--max_frames`: Maximum number of frames to use for each patient instance.

### ViT Model Configureation
- `--t_patch_size`: Temporal patch size for the model.
- `--num_frames`: Number of frames to use for training.
- `--pad_to_num_frames`: Pad the input to the specified number of frames.
- `--sep_pos_embed`: Use separate position embeddings for different modalities.
- `--cls_embed`: Use class embedding in the model.
- `--global_pool`: Use global pooling in the model.

### Extra Dataset-specific and Model-specific Configuration
In `scripts/cross-devices/finetune_aireadi_maestro2.sh` and `main_finetune_downstream_aireadi_correct_visit.py`, you can find the following dataset specific arguments.

In `main_finetune_downstream_singlefold_diffmodel.py`, you can find the following dataset specific arguments for how to use slivit model.

## Pre-training your 2D MAE model w/ flash attention
If in case you want to pre-train a 2D MAE model with flash attention, e.g., you want to keep training `RETFound` model on your own *en face* dataset, but with flash attention, you can use the following command:
```bash pretrain_command_oph_enface_flash_attn_retfound```

We provide simple structions on how to enable this script using Kermany dataset as an example.



# This repo contains the code for OCTCube fine-tuning.

### The code is adopted and modified based on the following references:
#### DeiT: https://github.com/facebookresearch/deit
#### BEiT: https://github.com/microsoft/unilm/tree/master/beit
#### MAE: https://github.com/facebookresearch/mae/tree/main
#### MAE_ST: https://github.com/facebookresearch/mae_st
#### RETFOUND_MAE: https://github.com/rmaphoh/RETFound_MAE
