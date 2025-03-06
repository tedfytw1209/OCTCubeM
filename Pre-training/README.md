# Pre-training of OCTCube

## Checklist:
- [] Add dataset examples to improve code-starting running experience.
- [x] Provide self-paced learning with 2D OCT dataset support, welcome to try it with external public 2D OCT dataset!
- [x] Release the pretrained model of OCTCube.
- [x] Dataset.md and README.md for better understanding of the dataset and the code (we are still refining it, welcome to any suggestions).
- [x] Upload code for pre-training OCTCube using 3D OCT data.



## Data preparation
For customized dataset used for pre-training, check the PatentDataset.py and PatientDataset_inhouse.py to see how to load the dataset. Specifically, check `Dataset.md` to find out more details of formatting and prepare pre-traiing dataset.

We provide our scripts that handles both 3D OCT and 2D OCT data. The 2D OCT data can be used for pre-training together, with a simple self-paced learning strategy. If you want to use this function and try it with external public 2D OCT dataset, we provide an example to prepare dataset, for more details, please refer to the `Dataset.md`.


## Run the pre-training
To pre-train OCTCube, please refer to commands in `scripts/`. As an example, to run the OCTCube pre-training, at OCTCube/Pre-training/scripts/, run 
``` source run_chunks_pretraining_vitl_oph_joint_flash_attn.sh ```


### Specific usage
Currently we support two ways of starting:
1. Starting from the pre-trained OCTCube.pth, to do this, check `run_chunks_pretraining_vitl_oph_joint_flash_attn.sh`, and replace the path with the correct path of OCTCube.pth.

2. Starting from the MAE pretrained on ImageNet, to do this check `run_chunks_pretraining_vitl_oph_joint_flash_attn_imagenet.sh`


## Pre-training Configuration Documentation

Below explains the parameters used in the script for pre-training models with a focus on vision transformers like ViT. Below are detailed descriptions of each parameter used in the script.

### General Parameters
- `BSZ`: Batch Size - Specifies the number of samples that will be processed before the model's internal parameters are updated.
- `INPUTSIZE`: Input Size - The size of the input images in pixels (assumes square images).
- `ACCUMSTEPS`: Gradient Accumulation Steps - Number of steps to accumulate gradients before updating model weights. Useful for handling larger batch sizes effectively on limited memory.
- `EPOCHS`: Number of Training Epochs - Total number of complete passes through the training dataset.
- `BLR`: Base Learning Rate - The starting learning rate for the optimizer.
- `RATIO`: Mask Ratio - Used in masked autoencoders, this specifies the fraction of the input tokens to mask for reconstruction tasks.

### Output Directories
- `OUTPUTDIR`: Output Directory - Base directory where the outputs of training (models, logs) will be saved.
- `SCHEDULER`: Scheduler Name - Descriptive name for the run configuration, incorporating various parameter values for easier identification of experiments.

### Training Script Execution
- `CUDA_VISIBLE_DEVICES`: Specifies which GPUs to use if multiple are available.
- `torchrun --nproc_per_node=4 --master_port=25680`: Command to run the script using PyTorch's distributed capabilities across 4 GPU nodes starting at port 25680.

### Training Parameters
- `--output_dir`: Directory to save output models and logs.
- `--log_dir`: Directory to save logs.
- `--batch_size`: Batch size for training.
- `--accum_iter`: Number of accumulation steps.
- `--epochs`: Number of epochs for training.
- `--blr`: Base learning rate.
- `--mask_ratio`: Ratio of input tokens to mask in masked autoencoder tasks.
- `--weight_decay`: Weight decay to prevent overfitting by penalizing large weights.
- `--num_workers`: Number of worker threads for loading data.
- `--num_frames`: Number of frames to consider for temporal modeling.
- `--t_patch_size`: Temporal patch size.
- `--pred_t_dim`: Dimension for predicting temporal features.
- `--input_size`: Dimension of square input images.
- `--warmup_epochs`: Number of warm-up epochs for learning rate scheduling.
- `--resume_type`: Type of resuming, if any; affects how training is resumed from a checkpoint. Default choices: ['training_latest', 'training_new', 'retfound', 'retfound_2_flash_attn', 'imagenet_2_flash_attn']
- `--model`: Model architecture specifier.
- `--eval_only`: Flag to indicate evaluation only mode.


### Self-paced learning for 2D OCT data
In the pre-training code, we further update the 3D MAE structure to support self-paced learning for 2D OCT data. For simplicity, we follow this [paper](https://papers.nips.cc/paper_files/paper/2014/hash/c60d060b946d6dd6145dcbad5c4ccf6f-Abstract.html) to implement the self-paced learning strategy. The main idea is to use the mean-squared error (MSE) loss to calculate the hardiness for 2D OCT data, and to pick up the top K% hardest samples with an monotonically increasing annealing strategy for the choice of K and mask ratio of the 2D OCT data to train the model. Below, we introduce related argument  to enable this function.


- `--batch_size_2d`: Batch size for 2D OCT data.
- `--mask_ratio_2d_min`: Minimum mask ratio for 2D OCT data.
- `--mask_ratio_2d_max`: Maximum mask ratio for 2D OCT data.
- `--K_min`: Minimum K% hardest samples for self-paced learning.
- `--K_max`: Maximum K% hardest samples for self-paced learning.
- `--epoch_offset`: Offset for the epoch count, used in annealing strategy.
- `--epoch_load_spl`: Epoch to load the self-paced learning model.
- `--load_spl_dir`: Directory to load the hardiness dictionary of the tracked 2D OCT images for the self-paced learning model.

To skip this self-paced learning function, set `--K_min` and `--K_max` to 0.
