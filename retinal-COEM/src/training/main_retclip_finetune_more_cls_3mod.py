# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import random
from datetime import datetime

import torchsummary
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from scipy.stats import pearsonr

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, get_context_length, get_vision_length, create_loaded_cv_ckpt
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train_retclip_finetune_more_cls_3mod import train_one_epoch, evaluate
from training.train_retclip_finetune_more_cls_3mod import plot_results

def compute_r2(y_true, y_pred):
    r_score, _ = pearsonr(y_true, y_pred)
    return r_score**2

def init_ongoing_best_metric_list_collection(data, num_classes, k_folds):
    def init_ongoing_best_metric_list(num_classes, k_folds):


        ongoing_best_val_metric_lists = [[[] for b in range(k_folds)] for _ in range(num_classes)]
        ongoing_best_val_metric_epoch_lists = [[-1 for b in range(k_folds)] for _ in range(num_classes)]
        ongoing_best_val_r2_lists = [[-1 for b in range(k_folds)] for _ in range(num_classes)]
        ongoing_best_val_dicted_result_combo_lists = [[[] for b in range(k_folds)] for _ in range(num_classes)]

        collected_list = {
            'ongoing_best_val_metric_lists': ongoing_best_val_metric_lists,
            'ongoing_best_val_metric_epoch_lists': ongoing_best_val_metric_epoch_lists,
            'ongoing_best_val_r2_lists': ongoing_best_val_r2_lists,
            'ongoing_best_val_dicted_result_combo_lists': ongoing_best_val_dicted_result_combo_lists
        }
        return collected_list

    ongoing_collection = {}
    if 'val' in data:
        ongoing_collection['val'] = init_ongoing_best_metric_list(num_classes, k_folds)
    if 'test' in data:
        num_test_sets = len(data['test'])
        ongoing_collection['test'] = [init_ongoing_best_metric_list(num_classes, k_folds) for _ in range(num_test_sets)]
        ongoing_collection['test_at_best_val'] = [init_ongoing_best_metric_list(num_classes, k_folds) for _ in range(num_test_sets)]
    if 'independent_test' in data:
        num_independent_test_sets = len(data['independent_test'])
        ongoing_collection['independent_test'] = [init_ongoing_best_metric_list(num_classes, k_folds) for _ in range(num_independent_test_sets)]
        ongoing_collection['independent_test_at_best_val'] = [init_ongoing_best_metric_list(num_classes, k_folds) for _ in range(num_independent_test_sets)]
    return ongoing_collection

def update_ongoing_best_metric_list_collection(ongoing_best_metric_list_collection, metric_list, setting, fold, dinfo_idx, num_classes,
    test_metric_list=None, test_independent_metric_list=None, dicted_result_combo_list=None, test_dicted_result_combo_list=None, independent_test_dicted_result_combo_list=None):
    '''
        metric_list: list of metrics for each epoch (num_classes), refers to a specific fold and epoch
    '''
    if setting == 'val':
        ongoing_best_val_metric_lists = ongoing_best_metric_list_collection['val']['ongoing_best_val_metric_lists']
        ongoing_best_val_metric_epoch_lists = ongoing_best_metric_list_collection['val']['ongoing_best_val_metric_epoch_lists']
        ongoing_best_val_r2_lists = ongoing_best_metric_list_collection['val']['ongoing_best_val_r2_lists']
        ongoing_best_val_dicted_result_combo_lists = ongoing_best_metric_list_collection['val']['ongoing_best_val_dicted_result_combo_lists']

    elif setting == 'test':
        ongoing_best_val_metric_lists = ongoing_best_metric_list_collection['test'][dinfo_idx]['ongoing_best_val_metric_lists']
        ongoing_best_val_metric_epoch_lists = ongoing_best_metric_list_collection['test'][dinfo_idx]['ongoing_best_val_metric_epoch_lists']
        ongoing_best_val_r2_lists = ongoing_best_metric_list_collection['test'][dinfo_idx]['ongoing_best_val_r2_lists']
        ongoing_best_val_dicted_result_combo_lists = ongoing_best_metric_list_collection['test'][dinfo_idx]['ongoing_best_val_dicted_result_combo_lists']
    elif setting == 'independent_test':
        ongoing_best_val_metric_lists = ongoing_best_metric_list_collection['independent_test'][dinfo_idx]['ongoing_best_val_metric_lists']
        ongoing_best_val_metric_epoch_lists = ongoing_best_metric_list_collection['independent_test'][dinfo_idx]['ongoing_best_val_metric_epoch_lists']
        ongoing_best_val_r2_lists = ongoing_best_metric_list_collection['independent_test'][dinfo_idx]['ongoing_best_val_r2_lists']
        ongoing_best_val_dicted_result_combo_lists = ongoing_best_metric_list_collection['independent_test'][dinfo_idx]['ongoing_best_val_dicted_result_combo_lists']

    newest_best_flag = [False for _ in range(num_classes)]
    current_epoch = len(metric_list) - 1
    for k in range(num_classes):
        metric_name = f'r2_{k}'
        previous_best_r2 = ongoing_best_val_r2_lists[k][fold]
        previous_best_epoch = ongoing_best_val_metric_epoch_lists[k][fold]
        newest_epoch_metric_list = metric_list[-1]
        newest_epoch_r2 = newest_epoch_metric_list[metric_name]
        newest_epoch = current_epoch
        if newest_epoch_r2 >= previous_best_r2:
            ongoing_best_val_metric_lists[k][fold] = [newest_epoch_metric_list]
            ongoing_best_val_metric_epoch_lists[k][fold] = newest_epoch
            ongoing_best_val_r2_lists[k][fold] = newest_epoch_r2
            ongoing_best_val_dicted_result_combo_lists[k][fold] = dicted_result_combo_list[-1]
            newest_best_flag[k] = True

            if setting == 'val':
                if 'test_at_best_val' in ongoing_best_metric_list_collection:
                    for test_idx, test_metric in enumerate(test_metric_list):
                        test_metric_name = f'r2_{k}'
                        ongoing_best_test_at_best_val_metric_lists = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_metric_lists']
                        ongoing_best_test_at_best_val_metric_epoch_lists = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists']
                        ongoing_best_test_at_best_val_r2_lists = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_r2_lists']
                        ongoing_best_test_at_best_val_dicted_result_lists = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_dicted_result_combo_lists']
                        # print(ongoing_best_test_at_best_val_metric_lists)
                        # print(ongoing_best_test_at_best_val_metric_epoch_lists)
                        # print(ongoing_best_test_at_best_val_r2_lists)
                        ongoing_best_test_at_best_val_metric_lists[k][fold] = [test_metric]
                        ongoing_best_test_at_best_val_metric_epoch_lists[k][fold] = current_epoch
                        ongoing_best_test_at_best_val_r2_lists[k][fold] = test_metric[test_metric_name]

                        test_dicted_result_combo_list_at_best_val = test_dicted_result_combo_list[test_idx][-1]
                        ongoing_best_test_at_best_val_dicted_result_lists[k][fold] = test_dicted_result_combo_list_at_best_val
                if 'independent_test_at_best_val' in ongoing_best_metric_list_collection:
                    for test_idx, independent_test_metric in enumerate(test_independent_metric_list):
                        independent_test_metric_name = f'r2_{k}'
                        ongoing_best_independent_test_at_best_val_metric_lists = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_metric_lists']
                        ongoing_best_independent_test_at_best_val_metric_epoch_lists = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists']
                        ongoing_best_independent_test_at_best_val_r2_lists = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_r2_lists']
                        ongoing_best_independent_test_at_best_val_dicted_result_lists = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_dicted_result_combo_lists']
                        # print(ongoing_best_independent_test_at_best_val_metric_lists)
                        # print(ongoing_best_independent_test_at_best_val_metric_epoch_lists)
                        # print(ongoing_best_independent_test_at_best_val_r2_lists)
                        ongoing_best_independent_test_at_best_val_metric_lists[k][fold] = [independent_test_metric]
                        ongoing_best_independent_test_at_best_val_metric_epoch_lists[k][fold] = current_epoch
                        ongoing_best_independent_test_at_best_val_r2_lists[k][fold] = independent_test_metric[independent_test_metric_name]

                        independent_test_dicted_result_combo_list_at_best_val = independent_test_dicted_result_combo_list[test_idx][-1]
                        ongoing_best_independent_test_at_best_val_dicted_result_lists[k][fold] = independent_test_dicted_result_combo_list_at_best_val

    return newest_best_flag




def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
        if args.multimodal_type == 'oct_faf_only':
            modality = 'fafonly'
            args.name += f"-modality_{modality}"
        if args.multimodal_type == 'oct3d_paired_faf_cls':
            modality = 'oct3d_faf_cls'
            modal_mapping = {'image': 'oct3d', 'text': 'faf'}
            args.name += f"-modality_{modality}"
            if args.cls_dataset:
                cls_dataset_type = args.cls_dataset_type
                args.name = str(cls_dataset_type) + '-' + args.name
        elif args.multimodal_type == 'oct3d_paired_ir_cls':
            modality = 'oct3d_ir_cls'
            modal_mapping = {'image': 'oct3d', 'text': 'ir'}
            args.name += f"-modality_{modality}"
            if args.cls_dataset:
                cls_dataset_type = args.cls_dataset_type
                args.name = str(cls_dataset_type) + '-' + args.name
        elif args.multimodal_type == 'oct3d_paired_faf_ir_cls':
            modality = 'oct3d_faf_ir_cls'
            modal_mapping = {'image': 'oct3d', 'text1': 'ir', 'text2': 'faf'}
            args.name += f"-modality_{modality}"
            if args.cls_dataset:
                cls_dataset_type = args.cls_dataset_type
                args.name = str(cls_dataset_type) + '-' + args.name
        if args.single_modality is not None:
            modal_name = modal_mapping[args.single_modality]
            args.name = f"sm-{modal_name}_" + args.name

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        # log_base_path = os.path.join(args.logs, args.name)
        # os.makedirs(log_base_path, exist_ok=True)
        # log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        if args.loaded_eval:
            ckpt_path_list, stored_expr_name = create_loaded_cv_ckpt(args)
            log_stored_path = os.path.join(args.logs, stored_expr_name)
            if args.model_selection_type == 'best_val':
                eval_name = args.model_selection_type
            else:
                eval_name = f'{args.model_selection_type}_{args.loaded_test_idx}'
            args.name = f'eval_{eval_name}_r2_{args.loaded_metric_idx}_' + args.name
            log_base_path = os.path.join(log_stored_path, args.name)
            log_filename = 'eval_out.log'
            args.logs = log_stored_path
        else:
            log_base_path = os.path.join(args.logs, args.name)
            log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_base_path = log_base_path

        print('Log base path:', log_base_path)
        # exit()
        os.makedirs(log_base_path, exist_ok=True)

        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    # initialize datasets
    print('Starting to get data')
    all_data = get_data(args, (None, None), epoch=0, \
                    tokenizer=get_tokenizer(args.model), context_length=get_context_length(args.model), \
                    vision_max_length=get_vision_length(args.model))
    assert len(all_data), 'At least one train or eval dataset must be specified.'
    assert len(all_data['val']) == 5
    assert args.k_folds == 5

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        if 'train' in all_data:
            args.train_sz = all_data["train"][0].dataloader.num_samples
        if args.val_data is not None or 'val' in data:
            args.val_sz = all_data["val"][0].dataloader.num_samples
        wandb_dir=os.path.join(args.logs, args.name, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            dir=wandb_dir,
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        logging.debug('Finished loading wandb.')

    best_val_metric_lists = [[] for _ in range(args.num_classes)]
    best_val_metric_epoch_lists = [[] for _ in range(args.num_classes)]
    best_val_r2_lists = [[] for _ in range(args.num_classes)]

    ongoing_best_metric_list_collection = init_ongoing_best_metric_list_collection(all_data, args.num_classes, args.k_folds)

    # Enable k_folds training
    for fold in range(args.k_folds):
        print(f'Starting fold {fold}')
        args.fold = fold
        data = {}
        data['train'] = all_data['train'][fold]
        data['val'] = all_data['val'][fold]
        if 'test' in all_data:
            data['test'] = all_data['test']
        if 'independent_test' in all_data:
            data['independent_test'] = all_data['independent_test']
        print(f'Train size: {len(data["train"].dataloader.dataset)}', f'Val size: {len(data["val"].dataloader.dataset)}')
        print(f'Test dataset number: {len(data["test"])}', f'Independent test dataset number: {len(data["independent_test"])}')
        print(f'Test dataset sizes: {[len(d.dataloader.dataset) for d in data["test"]]}', f'Independent test dataset sizes: {[len(d.dataloader.dataset) for d in data["independent_test"]]}')

        random_seed(args.seed, fold)

        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            pretrained_image=args.pretrained_image,
            image_mean=args.image_mean,
            image_std=args.image_std,
            args=args,
        )
        random_seed(args.seed, args.rank)

        if args.trace:
            model = trace_model(model, batch_size=args.batch_size, device=device)

        if args.lock_image:
            print('locking image tower')
            # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
            model.lock_image_tower(
                unlocked_groups=args.lock_image_unlocked_groups,
                freeze_bn_stats=args.lock_image_freeze_bn_stats)
        if args.lock_text:
            print('locking text tower')
            model.lock_text_tower(
                unlocked_layers=args.lock_text_unlocked_layers,
                freeze_layer_norm=args.lock_text_freeze_layer_norm)

        if args.grad_checkpointing:
            print('using grad checkpointing')
            model.set_grad_checkpointing()

        if is_master(args):
            logging.info("Model:")
            # FIXME: Un-comment this when we finish the adaptation
            # logging.info(f"{str(model)}")
            # torchsummary.summary(model)

            logging.info("Model Parameters Stats:")
            logging.info(f"  Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            logging.info(f"  Total non-trainable parameters = {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
            logging.info(f"  Vision trainable parameters = {sum(p.numel() for p in model.visual.parameters() if p.requires_grad)}")
            logging.info(f"  Vision non-trainable parameters = {sum(p.numel() for p in model.visual.parameters() if not p.requires_grad)}")

            # FIXME: Un-comment this when we finish the adaptation
            # # log the name and their shape of all model parameters
            # logging.info("Model Parameters List:")
            # for name, param in model.named_parameters():
            #     logging.info(f"  {name}: {param.shape}")

            logging.info("Training Parameters:")
            params_file = os.path.join(args.logs, args.name, "params.txt")
            with open(params_file, "w") as f:
                for name in sorted(vars(args)):
                    val = getattr(args, name)
                    logging.info(f"  {name}: {val}")
                    f.write(f"{name}: {val}\n")

        if args.distributed and not args.horovod:
            if args.use_bn_sync:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True, **ddp_args)

        # create optimizer and scaler
        optimizer = None
        scaler = None

        if (args.patient_dataset or args.combined_dataset or args.cls_dataset or args.dataset_type == "synthetic") and (not args.evaluate_only):
            assert not args.trace, 'Cannot train with traced model'
            print('Going right to load optimizer\n')

            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
            if args.horovod:
                optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)

            scaler = GradScaler() if args.precision == "amp" else None

        # optionally resume from a checkpoint
        start_epoch = 0
        if args.resume is not None:
            if os.path.isfile(args.resume):
                print('Loading checkpoint\n')
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    # resuming a train checkpoint w/ epoch and optimizer state

                    sd = checkpoint["state_dict"]
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    res = model.load_state_dict(sd, strict=False)
                    print(res)
                    if not args.not_load_epoch_when_resume:
                        start_epoch = checkpoint["epoch"]
                        if optimizer is not None:
                            optimizer.load_state_dict(checkpoint["optimizer"], strict=False)
                        if scaler is not None and 'scaler' in checkpoint:
                            scaler.load_state_dict(checkpoint['scaler'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
                else:
                    # loading a bare (model only) checkpoint for fine-tune or evaluation
                    model.load_state_dict(checkpoint)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))
                raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")


        # create scheduler if train
        scheduler = None
        if 'train' in data and optimizer is not None:
            total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

        # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
        args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
        writer = None
        if args.save_logs and args.tensorboard:
            assert tensorboard is not None, "Please install tensorboard."
            writer = tensorboard.SummaryWriter(args.tensorboard_path)

        if args.loaded_eval:
            print('Loading loaded eval checkpoint\n')
            ckpt_path = ckpt_path_list[fold]
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # model.load_state_dict(checkpoint["state_dict"])
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            res = model.load_state_dict(sd, strict=False)
            print(res)

            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {checkpoint['epoch']}) (fold {fold})")


        if 'train' not in data and args.evaluate_only:
            print('Evaluating model')
            evaluate(model, data, start_epoch, args, writer)
            continue
        # print('Exiting')
        # exit()

        total_train_batch_size = args.batch_size * args.accum_freq * args.world_size
        if is_master(args):
            logging.info("***** Running training *****")
            logging.info(f"  Num Epochs = {args.epochs}")
            logging.info(f"  Instantaneous batch size per device = {args.batch_size}")
            logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logging.info(f"  Gradient Accumulation steps = {args.accum_freq}")

        # Init recording for all datasets
        val_metric_list = []
        test_metric_list = [[] for _ in range(len(data['test']))]
        independent_test_metric_list = [[] for _ in range(len(data['independent_test']))]

        # Init recording of predicted results for all datasets
        val_predicted_results_list = []
        test_predicted_results_list = [[] for _ in range(len(data['test']))]
        independent_test_predicted_results_list = [[] for _ in range(len(data['independent_test']))]

        if args.loaded_eval:
            args.epochs = 1 # only evaluate once

        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logging.info(f'Start epoch {epoch}')

            if not args.loaded_eval:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)

            completed_epoch = epoch + 1

            if 'test' in data:
                print('Evaluating test')
                best_test_best_flag_list = []
                for test_idx, test_data in enumerate(data['test']):
                    print(f'Evaluating test {test_idx}')
                    test_metrics, test_dicted_results = evaluate(model, test_data, completed_epoch, args, writer, setting='test', dinfo_idx=0, return_prediction=True)
                    test_metric_list[test_idx].append(test_metrics)
                    test_predicted_results_list[test_idx].append(test_dicted_results)
                    # print('Before update', ongoing_best_metric_list_collection)
                    newest_best_test_flag = update_ongoing_best_metric_list_collection(ongoing_best_metric_list_collection, test_metric_list[test_idx], 'test', fold, test_idx, args.num_classes, dicted_result_combo_list=test_predicted_results_list[test_idx])
                    best_test_best_flag_list.append(newest_best_test_flag)
                    # print('after update', ongoing_best_metric_list_collection)
                    print(f'Newest best test flag: {newest_best_test_flag}')

            if 'independent_test' in data:
                print('Evaluating independent test')
                best_independent_test_best_flag_list = []
                for test_idx, independent_test_data in enumerate(data['independent_test']):
                    print(f'Evaluating independent test {test_idx}')
                    independent_test_metrics, independent_test_dicted_results = evaluate(model, independent_test_data, completed_epoch, args, writer, setting='independent_test', dinfo_idx=test_idx, return_prediction=True)
                    independent_test_metric_list[test_idx].append(independent_test_metrics)
                    independent_test_predicted_results_list[test_idx].append(independent_test_dicted_results)
                    # print('Before update', ongoing_best_metric_list_collection)
                    newest_best_independent_test_flag = update_ongoing_best_metric_list_collection(ongoing_best_metric_list_collection, independent_test_metric_list[test_idx], 'independent_test', fold, test_idx, args.num_classes, dicted_result_combo_list=independent_test_predicted_results_list[test_idx])
                    best_independent_test_best_flag_list.append(newest_best_independent_test_flag)
                    # print('after update', ongoing_best_metric_list_collection)
                    print(f'Newest best independent test flag: {newest_best_independent_test_flag}')




            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                print('Evaluating val')
                val_metrics, val_predicted_results = evaluate(model, data['val'], completed_epoch, args, writer, setting='val', return_prediction=True)
                # val_metrics = evaluate(model, data, completed_epoch, args, writer)
                val_metric_list.append(val_metrics)
                val_predicted_results_list.append(val_predicted_results)
                # print('Before update', ongoing_best_metric_list_collection)
                current_test_metric_list = [test_metric_list[i][-1] for i in range(len(test_metric_list))]
                current_independent_test_metric_list = [independent_test_metric_list[i][-1] for i in range(len(independent_test_metric_list))]
                newest_best_val_flag = update_ongoing_best_metric_list_collection(ongoing_best_metric_list_collection, val_metric_list, 'val', fold, 0, args.num_classes, test_metric_list=current_test_metric_list, test_independent_metric_list=current_independent_test_metric_list,
                    dicted_result_combo_list=val_predicted_results_list, test_dicted_result_combo_list=test_predicted_results_list, independent_test_dicted_result_combo_list=independent_test_predicted_results_list)
                # print('after update', ongoing_best_metric_list_collection)
                print(f'Newest best val flag: {newest_best_val_flag}')

            # Saving val checkpoints.
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                # Save val checkpoints if it is the best
                if not args.loaded_eval:
                    for k in range(args.num_classes):
                        val_folder_name = f'best_val/best_val_r2_{k}'
                        if not os.path.exists(os.path.join(args.checkpoint_path, val_folder_name)):
                            os.makedirs(os.path.join(args.checkpoint_path, val_folder_name))

                        if newest_best_val_flag[k]:
                            print(f'Saving best val checkpoint for r2_{k}')
                            torch.save(
                                checkpoint_dict,
                                os.path.join(args.checkpoint_path, f"{val_folder_name}/best_fold_{fold}.pt"),
                            )

                # Save test checkpoints if it is the best
                if 'test' in data and not args.loaded_eval:
                    for test_idx, test_data in enumerate(data['test']):
                        for k in range(args.num_classes):

                            test_folder_name = f'best_test/best_test_data_{test_idx}/r2_{k}'
                            if not os.path.exists(os.path.join(args.checkpoint_path, test_folder_name)):
                                os.makedirs(os.path.join(args.checkpoint_path, test_folder_name))
                            if best_test_best_flag_list[test_idx]:
                                print(f'Saving best test_{test_idx} checkpoint for r2_{k}')
                                torch.save(
                                    checkpoint_dict,
                                    os.path.join(args.checkpoint_path, f"{test_folder_name}/best_fold_{fold}.pt"),
                                )

                # Save independent test checkpoints if it is the best
                if 'independent_test' in data and not args.loaded_eval:
                    for test_idx, independent_test_data in enumerate(data['independent_test']):
                        for k in range(args.num_classes):
                            independent_test_folder_name = f'best_independent_test/best_independent_test_data_{test_idx}/r2_{k}'
                            if not os.path.exists(os.path.join(args.checkpoint_path, independent_test_folder_name)):
                                os.makedirs(os.path.join(args.checkpoint_path, independent_test_folder_name))
                            if best_independent_test_best_flag_list[test_idx]:
                                print(f'Saving best independent test_{test_idx} checkpoint for r2_{k}')
                                torch.save(
                                    checkpoint_dict,
                                    os.path.join(args.checkpoint_path, f"{independent_test_folder_name}/best_fold_{fold}.pt"),
                                )
                        if args.disable_extra_independent_test_save:
                            break

                # No use in fine-tuning
                # if args.save_most_recent:
                #     torch.save(
                #         checkpoint_dict,
                #         os.path.join(args.checkpoint_path, f"epoch_latest_fold_{fold}.pt"),
                #     )
                #                     )
                # if completed_epoch == args.epochs or (
                    # args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                # ) or (args.save_last_5 and completed_epoch >= args.epochs - 5):
                if not args.loaded_eval and args.save_last_5 and completed_epoch >= args.epochs - 5:
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}_fold_{fold}.pt"),
                    )
                    pass


        for k in range(args.num_classes):
            metric_name = f'r2_{k}'
            best_r2 = -1
            best_val_metric_list = best_val_metric_lists[k]
            best_val_metric_epoch_list = best_val_metric_epoch_lists[k]
            best_val_r2_list = best_val_r2_lists[k]
            for i, val_metrics in enumerate(val_metric_list):
                r2 = val_metrics[metric_name]
                if r2 > best_r2:
                    best_r2 = r2
                    best_epoch = i
            best_val_metric_list.append(val_metric_list[best_epoch])
            best_val_metric_epoch_list.append(best_epoch)
            best_val_r2_list.append(best_r2)
            print(f'Fold {fold} Best r2_{k}: {best_r2} at epoch {best_epoch}')
            if args.wandb and is_master(args):
                name = f'val_fold_{fold}'
                wandb.log({f'{name}/best_r2_{k}': best_r2, f'{name}/best_epoch_{k}': best_epoch})
                fold_name = f'val/fold_{fold}'
                wandb.log({f'{fold_name}_best_r2_{k}': best_r2, f'{fold_name}_best_epoch_{k}': best_epoch})
            if args.save_logs:
                result_name = "best_val_results.jsonl" if args.fold == -1 else f"best_val_results_fold_{args.fold}.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    f.write(f"best_r2_{k}: {best_r2} at epoch {best_epoch}\n")

            best_val_from_ongoing = ongoing_best_metric_list_collection['val']['ongoing_best_val_r2_lists'][k][fold]
            best_val_from_ongoing_epoch = ongoing_best_metric_list_collection['val']['ongoing_best_val_metric_epoch_lists'][k][fold]

            assert best_r2 == best_val_from_ongoing, f'Best r2 from ongoing collection {best_val_from_ongoing} is not the same as best r2 from evaluation {best_r2}'
            assert best_epoch == best_val_from_ongoing_epoch, f'Best epoch from ongoing collection {best_val_from_ongoing_epoch} is not the same as best epoch from evaluation {best_epoch}'

            if 'test' in data:

                for test_idx in range(len(data['test'])):
                    # get best test at val
                    best_test_metric_at_val = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    best_test_metric_epoch_at_val = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists'][k][fold]
                    best_test_r2_at_val = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'][k][fold]

                    if args.wandb and is_master(args):
                        print(f'Fold {fold} Best at_val_test_{test_idx}_r2_{k}: {best_r2} at epoch {best_epoch}')

                        name = f'test_{test_idx}_fold_{fold}'
                        wandb.log({f'{name}/r2_{k}_at_best_val': best_test_r2_at_val, f'{name}/epoch_{k}_at_best_val': best_test_metric_epoch_at_val})
                        fold_name = f'test_{test_idx}/fold_{fold}'
                        wandb.log({f'{fold_name}_r2_{k}_at_best_val': best_test_r2_at_val, f'{fold_name}_epoch_{k}_at_best_val': best_test_metric_epoch_at_val})

                    if args.save_logs:
                        result_name = f"test_{test_idx}_results_at_best_val.jsonl" if args.fold == -1 else f"test_{test_idx}_results_at_best_val_fold_{args.fold}.jsonl"
                        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                            f.write(f"at_best_val_test_{test_idx}_r2_{k}: {best_test_r2_at_val} at epoch {best_test_metric_epoch_at_val}\n")
                    # get best test
                    best_test_metric = ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    best_test_metric_epoch = ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_metric_epoch_lists'][k][fold]
                    best_test_r2 = ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    if args.wandb and is_master(args):
                        print(f'Fold {fold} Best test_{test_idx}_r2_{k}: {best_r2} at epoch {best_epoch}')

                        name = f'test_{test_idx}_fold_{fold}'
                        wandb.log({f'{name}/best_test_r2_{k}': best_test_r2, f'{name}/best_test_epoch_{k}': best_test_metric_epoch})
                        fold_name = f'test_{test_idx}/fold_{fold}'
                        wandb.log({f'{fold_name}_best_test_r2_{k}': best_test_r2, f'{fold_name}_best_test_epoch_{k}': best_test_metric_epoch})
                    if args.save_logs:
                        result_name = f"best_test_{test_idx}_results.jsonl" if args.fold == -1 else f"best_test_{test_idx}_results_fold_{args.fold}.jsonl"
                        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                            f.write(f"best_test_{test_idx}_r2_{k}: {best_test_r2} at epoch {best_test_metric_epoch}\n")

            if 'independent_test' in data:

                for test_idx in range(len(data['independent_test'])):
                    # get best independent test at val
                    best_independent_test_metric_at_val = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    best_independent_test_metric_epoch_at_val = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists'][k][fold]
                    best_independent_test_r2_at_val = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    if args.wandb and is_master(args):
                        print(f'Fold {fold} Best at_val_independent_test_{test_idx}_r2_{k}: {best_r2} at epoch {best_epoch}')

                        name = f'independent_test_{test_idx}_fold_{fold}'
                        wandb.log({f'{name}/r2_{k}_at_best_val': best_independent_test_r2_at_val, f'{name}/epoch_{k}_at_best_val': best_independent_test_metric_epoch_at_val})
                        fold_name = f'independent_test_{test_idx}/fold_{fold}'
                        wandb.log({f'{fold_name}_r2_{k}_at_best_val': best_independent_test_r2_at_val, f'{fold_name}_epoch_{k}_at_best_val': best_independent_test_metric_epoch_at_val})

                    if args.save_logs:
                        result_name = f"independent_test_{test_idx}_results_at_best_val.jsonl" if args.fold == -1 else f"independent_test_{test_idx}_results_at_best_val_fold_{args.fold}.jsonl"
                        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                            f.write(f"at_best_val_independent_test_{test_idx}_r2_{k}: {best_independent_test_r2_at_val} at epoch {best_independent_test_metric_epoch_at_val}\n")
                    # get best independent test
                    best_independent_test_metric = ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    best_independent_test_metric_epoch = ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_metric_epoch_lists'][k][fold]
                    best_independent_test_r2 = ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_r2_lists'][k][fold]
                    if args.wandb and is_master(args):
                        print(f'Fold {fold} Best independent_test_{test_idx}_r2_{k}: {best_r2} at epoch {best_epoch}')

                        name = f'independent_test_{test_idx}_fold_{fold}'
                        wandb.log({f'{name}/best_independent_test_r2_{k}': best_independent_test_r2, f'{name}/best_independent_test_epoch_{k}': best_independent_test_metric_epoch})
                        fold_name = f'independent_test_{test_idx}/fold_{fold}'
                        wandb.log({f'{fold_name}_best_independent_test_r2_{k}': best_independent_test_r2, f'{fold_name}_best_independent_test_epoch_{k}': best_independent_test_metric_epoch})
                    if args.save_logs:
                        result_name = f"best_independent_test_{test_idx}_results.jsonl" if args.fold == -1 else f"best_independent_test_{test_idx}_results_fold_{args.fold}.jsonl"
                        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                            f.write(f"best_independent_test_{test_idx}_r2_{k}: {best_independent_test_r2} at epoch {best_independent_test_metric_epoch}\n")



    avg_best_r2 = np.mean(best_val_r2_lists, axis=1)
    avg_best_epoch = np.mean(best_val_metric_epoch_lists, axis=1)
    std_best_r2 = np.std(best_val_r2_lists, axis=1)
    print(f'Average best r2: {avg_best_r2} with std {std_best_r2} at epoch {avg_best_epoch}')
    if args.wandb and is_master(args):
        for k in range(args.num_classes):
            wandb.log({f'val/avg_best_r2_{k}': avg_best_r2[k], f'val/avg_best_epoch_{k}': avg_best_epoch[k], f'val/std_best_r2_{k}': std_best_r2[k]})
        # wandb.log({'val/avg_best_r2': avg_best_r2, 'val/avg_best_epoch': avg_best_epoch, 'val/std_best_r2': std_best_r2})
    if args.save_logs:
        result_name = "best_val_results_avg.jsonl"
        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
            f.write(f"avg_best_r2: {avg_best_r2} with std {std_best_r2} at epoch {avg_best_epoch}\n")
            for k in range(args.num_classes):
                f.write(f"Now for class {k}\n")
                f.write(f"best_r2_{k}_list: {best_val_r2_lists[k]}\n")
                f.write(f"best_epoch_{k}_list: {best_val_metric_epoch_lists[k]}\n")
                f.write(f"best_metric_{k}_list: {best_val_metric_lists[k]}\n")

    # test and test at best val
    if 'test' in all_data:
        for test_idx in range(len(all_data['test'])):
            test_r2_at_best_val = np.mean(ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            test_epoch_at_best_val = np.mean(ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists'], axis=1)
            test_std_at_best_val = np.std(ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            print(f'Average test_{test_idx} r2 at best val: {test_r2_at_best_val} with std {test_std_at_best_val} at epoch {test_epoch_at_best_val}')
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    wandb.log({f'test_{test_idx}/avg_r2_{k}_at_best_val': test_r2_at_best_val[k], f'test_{test_idx}/avg_epoch_{k}_at_best_val': test_epoch_at_best_val[k], f'test_{test_idx}/std_r2_{k}_at_best_val': test_std_at_best_val[k]})
            if args.save_logs:
                result_name = f"test_{test_idx}_results_avg_at_best_val.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    f.write(f"avg_r2_at_best_val: {test_r2_at_best_val} with std {test_std_at_best_val} at epoch {test_epoch_at_best_val}\n")
                    for k in range(args.num_classes):
                        f.write(f"Now for class {k}\n")
                        f.write(f"best_r2_{k}_list: {ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'][k]}\n")
                        f.write(f"best_epoch_{k}_list: {ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists'][k]}\n")
                        f.write(f"best_metric_{k}_list: {ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_metric_lists'][k]}\n")
            # now best test
            best_test_r2 = np.mean(ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            best_test_epoch = np.mean(ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_metric_epoch_lists'], axis=1)
            best_test_std = np.std(ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            print(f'Average best test_{test_idx} r2: {best_test_r2} with std {best_test_std} at epoch {best_test_epoch}')
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    wandb.log({f'test_{test_idx}/avg_best_r2_{k}': best_test_r2[k], f'test_{test_idx}/avg_best_epoch_{k}': best_test_epoch[k], f'test_{test_idx}/std_best_r2_{k}': best_test_std[k]})
            if args.save_logs:
                result_name = f"best_test_{test_idx}_results_avg.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    f.write(f"avg_best_r2: {best_test_r2} with std {best_test_std} at epoch {best_test_epoch}\n")
                    for k in range(args.num_classes):
                        f.write(f"Now for class {k}\n")
                        f.write(f"best_r2_{k}_list: {ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_r2_lists'][k]}\n")
                        f.write(f"best_epoch_{k}_list: {ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_metric_epoch_lists'][k]}\n")
                        f.write(f"best_metric_{k}_list: {ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_metric_lists'][k]}\n")
            # get ensemble test_at_best_val
            test_dicted_result_combo_list = ongoing_best_metric_list_collection['test_at_best_val'][test_idx]['ongoing_best_val_dicted_result_combo_lists']
            gt_labels = test_dicted_result_combo_list[0][0]['original_labels']
            true_idxes = test_dicted_result_combo_list[0][0]['original_true_idx']
            avg_ensemble_logits_classes_list = [np.zeros_like(test_dicted_result_combo_list[0][0]['original_logits']) for _ in range(args.num_classes)]
            print(len(avg_ensemble_logits_classes_list), avg_ensemble_logits_classes_list[0].shape)
            print(test_dicted_result_combo_list[0][0]['original_logits'].shape, len(test_dicted_result_combo_list), len(test_dicted_result_combo_list[0]))

            avg_ensemble_metric_classes_list = [[[] for _ in range(args.num_classes)] for _ in range(args.num_classes)]
            for k in range(args.num_classes):
                avg_ensemble_logits = avg_ensemble_logits_classes_list[k]
                test_dicted_result_folds = test_dicted_result_combo_list[k]
                for fd, test_dicted_result in enumerate(test_dicted_result_folds):
                    avg_ensemble_logits += test_dicted_result['original_logits']
                avg_ensemble_logits /= args.k_folds
                for inner_cls in range(args.num_classes):
                    mse = np.mean((avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls])**2)
                    mae = np.mean(np.abs(avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls]))
                    r2 = compute_r2(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])
                    pearson = pearsonr(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])[0]
                    ens_res_dict = {'r2': r2, 'pearson': pearson, 'mse': mse, 'mae': mae}
                    avg_ensemble_metric_classes_list[k][inner_cls] = ens_res_dict
            # upload to wandb
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    for inner_cls in range(args.num_classes):
                        ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                        for key, val in ens_res_dict.items():
                            wandb.log({f'test_{test_idx}/avg_ensemble_test_at_best_val_sel_{k}_cls_{inner_cls}_key_{key}': val})
            # save to logs
            if args.save_logs:
                result_name = f"test_at_best_val_{test_idx}_ensemble_results_avg.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    for k in range(args.num_classes):
                        for inner_cls in range(args.num_classes):
                            ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                            f.write(f"avg_ensemble_sel_{k}_cls_{inner_cls}_res: {ens_res_dict}\n")
                # also save the avg_enemble_logits_classes_list
                results_to_be_saved = {
                    'avg_ensemble_logits_classes_list': avg_ensemble_logits_classes_list,
                    'gt_labels': gt_labels,
                    'true_idxes': true_idxes
                }

                with open(os.path.join(args.checkpoint_path, f"test_at_best_val_{test_idx}_ensemble_results_avg.pkl"), "wb") as f:
                    pickle.dump(results_to_be_saved, f)
                # plot the scatter plot
                plot_folder_name = f"test_at_best_val_{test_idx}_ensemble_results_plots"
                if not os.path.exists(os.path.join(args.checkpoint_path, plot_folder_name)):
                    os.makedirs(os.path.join(args.checkpoint_path, plot_folder_name))
                for k in range(args.num_classes):
                    plot_logits = avg_ensemble_logits_classes_list[k]
                    plot_gt_labels = gt_labels
                    fig_list = plot_results(args, plot_gt_labels, plot_logits)
                    for i, fig in enumerate(fig_list):
                        fig.savefig(os.path.join(args.checkpoint_path, plot_folder_name, f"ensemble_test_at_best_val_{test_idx}_class_{k}_plot_{i}.png"))
                        wandb.log({f'test_{test_idx}_ensemble/avg_ensemble_test_{test_idx}_at_best_val_sel_{k}_cls_{i}': wandb.Image(fig)})


            # get ensemble best test
            test_dicted_result_combo_list = ongoing_best_metric_list_collection['test'][test_idx]['ongoing_best_val_dicted_result_combo_lists']
            gt_labels = test_dicted_result_combo_list[0][0]['original_labels']
            true_idxes = test_dicted_result_combo_list[0][0]['original_true_idx']
            avg_ensemble_logits_classes_list = [np.zeros_like(test_dicted_result_combo_list[0][0]['original_logits']) for _ in range(args.num_classes)]
            avg_ensemble_metric_classes_list = [[[] for _ in range(args.num_classes)] for _ in range(args.num_classes)]
            for k in range(args.num_classes):
                avg_ensemble_logits = avg_ensemble_logits_classes_list[k]
                test_dicted_result_folds = test_dicted_result_combo_list[k]
                for fd, test_dicted_result in enumerate(test_dicted_result_folds):
                    avg_ensemble_logits += test_dicted_result['original_logits']
                avg_ensemble_logits /= args.k_folds
                for inner_cls in range(args.num_classes):
                    mse = np.mean((avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls])**2)
                    mae = np.mean(np.abs(avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls]))
                    r2 = compute_r2(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])
                    pearson = pearsonr(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])[0]
                    ens_res_dict = {'r2': r2, 'pearson': pearson, 'mse': mse, 'mae': mae}
                    avg_ensemble_metric_classes_list[k][inner_cls] = ens_res_dict
            # upload to wandb
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    for inner_cls in range(args.num_classes):
                        ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                        for key, val in ens_res_dict.items():
                            wandb.log({f'test_{test_idx}/avg_ensemble_best_test_sel_{k}_cls_{inner_cls}_key_{key}': val})
            # save to logs
            if args.save_logs:
                result_name = f"best_test_{test_idx}_ensemble_results_avg.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    for k in range(args.num_classes):
                        for inner_cls in range(args.num_classes):
                            ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                            f.write(f"avg_ensemble_sel_{k}_cls_{inner_cls}_res: {ens_res_dict}\n")
                # also save the avg_enemble_logits_classes_list
                results_to_be_saved = {
                    'avg_ensemble_logits_classes_list': avg_ensemble_logits_classes_list,
                    'gt_labels': gt_labels,
                    'true_idxes': true_idxes
                }
                with open(os.path.join(args.checkpoint_path, f"best_test_{test_idx}_ensemble_results_avg.pkl"), "wb") as f:
                    pickle.dump(results_to_be_saved, f)

                # plot the scatter plot
                plot_folder_name = f"best_test_{test_idx}_ensemble_results_plots"
                if not os.path.exists(os.path.join(args.checkpoint_path, plot_folder_name)):
                    os.makedirs(os.path.join(args.checkpoint_path, plot_folder_name))
                for k in range(args.num_classes):
                    plot_logits = avg_ensemble_logits_classes_list[k]
                    plot_gt_labels = gt_labels
                    fig_list = plot_results(args, plot_gt_labels, plot_logits)
                    for i, fig in enumerate(fig_list):
                        fig.savefig(os.path.join(args.checkpoint_path, plot_folder_name, f"ensemble_best_test_{test_idx}_class_{k}_plot_{i}.png"))
                        wandb.log({f'test_{test_idx}_ensemble/avg_ensemble_best_test_{test_idx}_sel_{k}_cls_{i}': wandb.Image(fig)})



    # now independent test
    if 'independent_test' in all_data:
        for test_idx in range(len(all_data['independent_test'])):
            independent_test_r2_at_best_val = np.mean(ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            independent_test_epoch_at_best_val = np.mean(ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists'], axis=1)
            independent_test_std_at_best_val = np.std(ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            print(f'Average independent test_{test_idx} r2 at best val: {independent_test_r2_at_best_val} with std {independent_test_std_at_best_val} at epoch {independent_test_epoch_at_best_val}')
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    wandb.log({f'independent_test_{test_idx}/avg_r2_{k}_at_best_val': independent_test_r2_at_best_val[k], f'independent_test_{test_idx}/avg_epoch_{k}_at_best_val': independent_test_epoch_at_best_val[k], f'independent_test_{test_idx}/std_r2_{k}_at_best_val': independent_test_std_at_best_val[k]})
            if args.save_logs:
                result_name = f"independent_test_{test_idx}_results_avg_at_best_val.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    f.write(f"avg_r2_at_best_val: {independent_test_r2_at_best_val} with std {independent_test_std_at_best_val} at epoch {independent_test_epoch_at_best_val}\n")
                    for k in range(args.num_classes):
                        f.write(f"Now for class {k}\n")
                        f.write(f"best_r2_{k}_list: {ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_r2_lists'][k]}\n")
                        f.write(f"best_epoch_{k}_list: {ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_metric_epoch_lists'][k]}\n")
                        f.write(f"best_metric_{k}_list: {ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_metric_lists'][k]}\n")
            # now best independent test
            best_independent_test_r2 = np.mean(ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            best_independent_test_epoch = np.mean(ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_metric_epoch_lists'], axis=1)
            best_independent_test_std = np.std(ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_r2_lists'], axis=1)
            print(f'Average best independent test_{test_idx} r2: {best_independent_test_r2} with std {best_independent_test_std} at epoch {best_independent_test_epoch}')
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    wandb.log({f'independent_test_{test_idx}/avg_best_r2_{k}': best_independent_test_r2[k], f'independent_test_{test_idx}/avg_best_epoch_{k}': best_independent_test_epoch[k], f'independent_test_{test_idx}/std_best_r2_{k}': best_independent_test_std[k]})
            if args.save_logs:
                result_name = f"best_independent_test_{test_idx}_results_avg.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    f.write(f"avg_best_r2: {best_independent_test_r2} with std {best_independent_test_std} at epoch {best_independent_test_epoch}\n")
                    for k in range(args.num_classes):
                        f.write(f"Now for class {k}\n")
                        f.write(f"best_r2_{k}_list: {ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_r2_lists'][k]}\n")
                        f.write(f"best_epoch_{k}_list: {ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_metric_epoch_lists'][k]}\n")
                        f.write(f"best_metric_{k}_list: {ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_metric_lists'][k]}\n")

            # get ensemble independent_test_at_best_val
            independent_test_dicted_result_combo_list = ongoing_best_metric_list_collection['independent_test_at_best_val'][test_idx]['ongoing_best_val_dicted_result_combo_lists']
            gt_labels = independent_test_dicted_result_combo_list[0][0]['original_labels']
            true_idxes = independent_test_dicted_result_combo_list[0][0]['original_true_idx']
            avg_ensemble_logits_classes_list = [np.zeros_like(independent_test_dicted_result_combo_list[0][0]['original_logits']) for _ in range(args.num_classes)]
            avg_ensemble_metric_classes_list = [[[] for _ in range(args.num_classes)] for _ in range(args.num_classes)]
            for k in range(args.num_classes):
                avg_ensemble_logits = avg_ensemble_logits_classes_list[k]
                independent_test_dicted_result_folds = independent_test_dicted_result_combo_list[k]
                for fd, independent_test_dicted_result in enumerate(independent_test_dicted_result_folds):
                    avg_ensemble_logits += independent_test_dicted_result['original_logits']
                avg_ensemble_logits /= args.k_folds
                for inner_cls in range(args.num_classes):
                    mse = np.mean((avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls])**2)
                    mae = np.mean(np.abs(avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls]))
                    r2 = compute_r2(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])
                    pearson = pearsonr(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])[0]
                    ens_res_dict = {'r2': r2, 'pearson': pearson, 'mse': mse, 'mae': mae}
                    avg_ensemble_metric_classes_list[k][inner_cls] = ens_res_dict
            # upload to wandb
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    for inner_cls in range(args.num_classes):
                        ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                        for key, val in ens_res_dict.items():
                            wandb.log({f'independent_test_{test_idx}/avg_ensemble_test_at_best_val_sel_{k}_cls_{inner_cls}_key_{key}': val})
            # save to logs
            if args.save_logs:
                result_name = f"independent_test_at_best_val_{test_idx}_ensemble_results_avg.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    for k in range(args.num_classes):
                        for inner_cls in range(args.num_classes):
                            ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                            f.write(f"avg_ensemble_sel_{k}_cls_{inner_cls}_res: {ens_res_dict}\n")
                # also save the avg_enemble_logits_classes_list
                results_to_be_saved = {
                    'avg_ensemble_logits_classes_list': avg_ensemble_logits_classes_list,
                    'gt_labels': gt_labels,
                    'true_idxes': true_idxes
                }
                with open(os.path.join(args.checkpoint_path, f"independent_test_at_best_val_{test_idx}_ensemble_results_avg.pkl"), "wb") as f:
                    pickle.dump(results_to_be_saved, f)
                # plot the scatter plot
                plot_folder_name = f"independent_test_at_best_val_{test_idx}_ensemble_results_plots"
                if not os.path.exists(os.path.join(args.checkpoint_path, plot_folder_name)):
                    os.makedirs(os.path.join(args.checkpoint_path, plot_folder_name))
                for k in range(args.num_classes):
                    plot_logits = avg_ensemble_logits_classes_list[k]
                    plot_gt_labels = gt_labels
                    fig_list = plot_results(args, plot_gt_labels, plot_logits)
                    for i, fig in enumerate(fig_list):
                        fig.savefig(os.path.join(args.checkpoint_path, plot_folder_name, f"ensemble_independent_test_at_best_val_{test_idx}_class_{k}_plot_{i}.png"))
                        wandb.log({f'independent_test_{test_idx}_ensemble/avg_ensemble_independent_test_{test_idx}_at_best_val_sel_{k}_cls_{i}': wandb.Image(fig)})


            # get ensemble best independent test
            independent_test_dicted_result_combo_list = ongoing_best_metric_list_collection['independent_test'][test_idx]['ongoing_best_val_dicted_result_combo_lists']
            gt_labels = independent_test_dicted_result_combo_list[0][0]['original_labels']
            true_idxes = independent_test_dicted_result_combo_list[0][0]['original_true_idx']
            avg_ensemble_logits_classes_list = [np.zeros_like(independent_test_dicted_result_combo_list[0][0]['original_logits']) for _ in range(args.num_classes)]
            avg_ensemble_metric_classes_list = [[[] for _ in range(args.num_classes)] for _ in range(args.num_classes)]
            for k in range(args.num_classes):
                avg_ensemble_logits = avg_ensemble_logits_classes_list[k]
                independent_test_dicted_result_folds = independent_test_dicted_result_combo_list[k]
                for fd, independent_test_dicted_result in enumerate(independent_test_dicted_result_folds):
                    avg_ensemble_logits += independent_test_dicted_result['original_logits']
                avg_ensemble_logits /= args.k_folds
                for inner_cls in range(args.num_classes):
                    mse = np.mean((avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls])**2)
                    mae = np.mean(np.abs(avg_ensemble_logits[:, inner_cls] - gt_labels[:, inner_cls]))
                    r2 = compute_r2(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])
                    pearson = pearsonr(gt_labels[:, inner_cls], avg_ensemble_logits[:, inner_cls])[0]
                    ens_res_dict = {'r2': r2, 'pearson': pearson, 'mse': mse, 'mae': mae}
                    avg_ensemble_metric_classes_list[k][inner_cls] = ens_res_dict
            # upload to wandb
            if args.wandb and is_master(args):
                for k in range(args.num_classes):
                    for inner_cls in range(args.num_classes):
                        ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                        for key, val in ens_res_dict.items():
                            wandb.log({f'independent_test_{test_idx}/avg_ensemble_best_test_sel_{k}_cls_{inner_cls}_key_{key}': val})
            # save to logs
            if args.save_logs:
                result_name = f"best_independent_test_{test_idx}_ensemble_results_avg.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    for k in range(args.num_classes):
                        for inner_cls in range(args.num_classes):
                            ens_res_dict = avg_ensemble_metric_classes_list[k][inner_cls]
                            f.write(f"avg_ensemble_sel_{k}_cls_{inner_cls}_res: {ens_res_dict}\n")
                # also save the avg_enemble_logits_classes_list
                results_to_be_saved = {
                    'avg_ensemble_logits_classes_list': avg_ensemble_logits_classes_list,
                    'gt_labels': gt_labels,
                    'true_idxes': true_idxes
                }
                with open(os.path.join(args.checkpoint_path, f"best_independent_test_{test_idx}_ensemble_results_avg.pkl"), "wb") as f:
                    pickle.dump(results_to_be_saved, f)
                # plot the scatter plot
                plot_folder_name = f"best_independent_test_{test_idx}_ensemble_results_plots"
                if not os.path.exists(os.path.join(args.checkpoint_path, plot_folder_name)):
                    os.makedirs(os.path.join(args.checkpoint_path, plot_folder_name))
                for k in range(args.num_classes):
                    plot_logits = avg_ensemble_logits_classes_list[k]
                    plot_gt_labels = gt_labels
                    fig_list = plot_results(args, plot_gt_labels, plot_logits)
                    for i, fig in enumerate(fig_list):
                        fig.savefig(os.path.join(args.checkpoint_path, plot_folder_name, f"ensemble_best_independent_test_{test_idx}_class_{k}_plot_{i}.png"))
                        wandb.log({f'independent_test_{test_idx}_ensemble/avg_ensemble_best_independent_test_{test_idx}_sel_{k}_cls_{i}': wandb.Image(fig)})



    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])
