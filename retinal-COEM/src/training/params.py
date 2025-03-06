import argparse
import os

home_directory = os.getenv('HOME') + '/'


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def parse_input_size(value):
    # Attempt to evaluate the input if it's given as tuple
    try:
        # Convert input to a tuple if it's in form "x,y"
        if ',' in value:
            size = tuple(map(int, value.split(',')))
            if len(size) == 2:
                return size
            else:
                raise argparse.ArgumentTypeError("Tuple must have exactly two integers.")
        # Otherwise, handle it as an integer
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid input size value: {value}")

def parse_args(args):
    # print('are we here?')
    parser = argparse.ArgumentParser()

    # Accelerate training arguments (by disabling extra independent test sets)
    parser.add_argument('--disable_extra_independent_test_save', default=False, action='store_true', help='disable extra independent test sets saving')



    # loaded_model arguments
    example_model_ckpt = '2e-5_GAGrowth_stage2_mm-OCT-FAF_full-ft'
    parser.add_argument('--expr_dir', default=expr_dir, type=str, help='expr directory')
    parser.add_argument('--loaded_eval', default=False, action='store_true', help='loaded eval')
    parser.add_argument('--loaded_expr_name', default=example_model_ckpt, type=str, help='loaded expr name')
    parser.add_argument('--model_selection_type', default='best_val', type=str, choices=['best_val', 'best_test', 'best_independent_test'], help='model selection type')
    parser.add_argument('--loaded_test_idx', default=0, type=int, help='loaded test idx')
    parser.add_argument('--loaded_metric_idx', default=1, type=int, help='loaded metric idx')
    # persistent_dataset
    parser.add_argument('--dup_oct_3_channels', default=False, action='store_true', help='duplicate oct 3 channelsm, only used for RETFound model')
    parser.add_argument('--enable_3mod_training', default=False, action='store_true', help='enable 3mod training')
    parser.add_argument('--persistent_dataset_dir', default=persistent_dataset_dir, type=str, help='persistent dataset directory')
    parser.add_argument('--current_dir', default=current_dir, type=str, help='current directory')
    parser.add_argument('--combined_dataset', default=False, action='store_true', help='use combined dataset, for combined dataset, newly added') 
    parser.add_argument('--cls_dataset', default=False, action='store_true', help='use cls dataset')
    parser.add_argument('--multimodal_type', default='oct_ir', type=str, choices=['default', 'oct_ir', 'oct_faf_only', 'oct_faf_all', 'oct3d_paired_faf_cls', 'oct3d_paired_ir_cls', 'oct_faf_ir', 'oct3d_paired_faf_ir_cls'], help='multimodal type')

    parser.add_argument('--save_last_5', default=False, action='store_true', help='save last 5 frames')
    # parser.add_argument('--evaluate_only_patient_dataset', default=False, action='store_true', help='evaluate only patient dataset')
    # parser.add_argument('--evaluate_only_proximaB_dataset', default=False, action='store_true', help='evaluate only proximaB dataset')
    parser.add_argument('--not_load_epoch_when_resume', default=False, action='store_true', help='not load epoch when resume') 
    parser.add_argument('--use_faf_all', default=False, action='store_true', help='use faf_all dataset')

    parser.add_argument('--cls_dataset_type', default='GAGrowth', type=str, choices=[
    'GAGrowth', 'BCVA_and_GAA'], help='load cls_dataset type')
    parser.add_argument('--k_folds', default=5, type=int, help='number of folds')
    parser.add_argument('--fold', default=-1, type=int, help='fold, only used with cls_dataset and cv')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--single_modality', default=None, type=str, help='image or text, single modality, only used for single modality ablative study, need to specify the modality') # choices=['image', 'text', 'text1', 'text2']
    parser.add_argument('--enable_independent_test', default=False, action='store_true', help='enable independent test')
    # AIREADI
    parser.add_argument('--aireadi_only_include_pair', default=False, action='store_true', help='location of the aireadi dataset')
    parser.add_argument('--aireadi_location', default='Macula', type=str, help='location of the aireadi dataset')
    parser.add_argument('--aireadi_device', default='Spectralis', type=str, help='device of the aireadi dataset')
    parser.add_argument('--aireadi_split', default='All', type=str, help='split of the aireadi dataset')
    parser.add_argument('--aireadi_pre_patient_cohort', default='All_have', type=str, help='pre_patient_cohort of the aireadi dataset')
    parser.add_argument('--aireadi_abnormal_file_tsv', default=None, type=str, help='abnormal abnormal file tsv')
    parser.add_argument('--shift_mean_std', default=False, action='store_true', help='shift mean and std')
    parser.add_argument('--aireadi_normalize_retfound', default=False, action='store_true', help='normalize aireadi dataset with retfound mean and std')
    parser.add_argument('--dataset_mode', default='frame', type=str, help='dataset mode')

    parser.add_argument('--return_patient_id', default=False, action='store_true', help='return patient id')
    parser.add_argument('--evaluate_all', default=False, action='store_true', help='evaluate all')
    parser.add_argument('--return_metainfo', default=False, action='store_true', help='return metainfo')
    parser.add_argument('--save_retrieval_results', default=False, action='store_true', help='save retrieval results')
    parser.add_argument('--patient_dataset', default=True, action='store_true', help='Use patient dataset')
    parser.add_argument('--evaluate_only', default=False, action='store_true', help='evaluate only')
    parser.add_argument('--normalize_dataset', default=False, action='store_true', help='normalize dataset, used for baseline2 model')
    parser.add_argument('--input_size', default=224, type=parse_input_size, help='oct images input size')
    parser.add_argument('--input_size_ir', default=224, type=int, help='enface images input size')
    parser.add_argument('--dataset_type', default='3D', type=str, choices=['Center2D', '3D'], help='dataset type')

    parser.add_argument('--downsample_normal', default=False, action='store_true', help='downsample normal cases')
    parser.add_argument('--downsample_normal_factor', default=10, type=int, help='downsample normal cases by a factor')
    parser.add_argument('--same_3_frames', default=False, action='store_true', help='use the same 3 frames to mock 1 frame for 3D spatio-temporal model')

    parser.add_argument('--variable_joint', default=False, action='store_true', help='use variable joint attention')
    parser.add_argument('--high_res_num_frames', default=30, type=int, help='number of high resolution frames')

    parser.add_argument('--load_non_flash_attn_to_flash_attn', default=False, action='store_true', help='use focal loss')

    parser.add_argument("--num_frames", default=-1, type=int)
    parser.add_argument("--pad_to_num_frames", default=False, action="store_true")
    parser.add_argument("--transform_type", default="frame_2D", type=str, choices=["frame_2D", "monai_3D"])
    parser.add_argument("--color_mode", default="rgb", type=str, choices=["rgb", "gray"])
    parser.add_argument('--disease', default='AMD', type=str, choices=['AMD', 'DME', 'POG', 'ODR', 'PM', 'CRO', 'RN', 'VD'], help='Disease type for the dataset (only for binary_cls task_mode)')
    parser.add_argument('--data_path', default=home_directory + '/Ophthal/', type=str, help='dataset path')

    parser.add_argument('--task_mode', default='multi_label', type=str, choices=['binary_cls', 'multi_cls', 'multi_label'], help='Task mode for the dataset')
    parser.add_argument('--patient_id_list_dir', default='multi_cls_expr_10x_0315/', type=str, help='patient id list dir')
    parser.add_argument('--split_path', default=None, type=str, help='split path storing the train/val/test split of patient files')
    parser.add_argument('--few_shot', default=False, action='store_true', help='finetune from checkpoint')
    parser.add_argument("--smaller_temporal_crop", default='interp', type=str, choices=['interp', 'crop'], help='interpolation type for temporal position embedding')
    parser.add_argument("--prefetch_factor", default=2, type=int, help='prefetch factor')

    parser.add_argument(
        "--train-data",
        type=str,
        default="",
        help="Path to file(s) with training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="",
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "tsv", "synthetic", "auto"],
        default="csv",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="encoded_image",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="caption",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--csv-label-key",
        type=str,
        default="patient",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size per GPU."
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=None, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=2, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--max-samples-per-epoch", type=int, default=None, help="How many samples to use per epoch."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PubMedBERT_512-LongNet-Layers-12-Dim-256-MAE",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        "--correct-label",
        default=1,
        type=int,
        help="Whether to consider that there may be multiple slides corresponding to one report",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=True,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=True,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip-ret',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=True,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )


    args = parser.parse_args(args)


    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
