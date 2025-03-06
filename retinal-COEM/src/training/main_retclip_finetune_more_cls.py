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
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, get_context_length, get_vision_length
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train_retclip_finetune_more_cls import train_one_epoch, evaluate


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
        if args.single_modality is not None:
            modal_name = modal_mapping[args.single_modality]
            args.name = f"sm-{modal_name}_" + args.name

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
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
    # Enable k_folds training
    for fold in range(args.k_folds):
        print(f'Starting fold {fold}')
        args.fold = fold
        data = {}
        data['train'] = all_data['train'][fold]
        data['val'] = all_data['val'][fold]

        print(f'Train size: {len(data["train"].dataloader.dataset)}', f'Val size: {len(data["val"].dataloader.dataset)}')


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
        # exit()

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
                raise FileNotFoundError(f"no checkpoint found at '{args.resume}'")


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
        val_metric_list = []
        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logging.info(f'Start epoch {epoch}')

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)
            completed_epoch = epoch + 1

            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                val_metrics = evaluate(model, data, completed_epoch, args, writer)
                val_metric_list.append(val_metrics)

            # Saving checkpoints.
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                ) or (args.save_last_5 and completed_epoch >= args.epochs - 5):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}_fold_{fold}.pt"),
                    )
                if args.save_most_recent:
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_latest_fold_{fold}.pt"),
                    )
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
                result_name = "best_results.jsonl" if args.fold == -1 else f"best_results_fold_{args.fold}.jsonl"
                with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
                    f.write(f"best_r2_{k}: {best_r2} at epoch {best_epoch}\n")


    avg_best_r2 = np.mean(best_val_r2_lists, axis=1)
    avg_best_epoch = np.mean(best_val_metric_epoch_lists, axis=1)
    std_best_r2 = np.std(best_val_r2_lists, axis=1)
    print(f'Average best r2: {avg_best_r2} with std {std_best_r2} at epoch {avg_best_epoch}')
    if args.wandb and is_master(args):
        for k in range(args.num_classes):
            wandb.log({f'val/avg_best_r2_{k}': avg_best_r2[k], f'val/avg_best_epoch_{k}': avg_best_epoch[k], f'val/std_best_r2_{k}': std_best_r2[k]})
        # wandb.log({'val/avg_best_r2': avg_best_r2, 'val/avg_best_epoch': avg_best_epoch, 'val/std_best_r2': std_best_r2})
    if args.save_logs:
        result_name = "best_results_avg.jsonl"
        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
            f.write(f"avg_best_r2: {avg_best_r2} with std {std_best_r2} at epoch {avg_best_epoch}\n")
            for k in range(args.num_classes):
                f.write(f"Now for class {k}\n")
                f.write(f"best_r2_{k}_list: {best_val_r2_lists[k]}\n")
                f.write(f"best_epoch_{k}_list: {best_val_metric_epoch_lists[k]}\n")
                f.write(f"best_metric_{k}_list: {best_val_metric_lists[k]}\n")


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
