# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#from cProfile import label
import json
import logging
import math
import os
import time
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    wandb = None

from sklearn.metrics import recall_score
from open_clip import ClipLoss, get_cast_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error




def compute_r2(y_true, y_pred):
    r_score, _ = pearsonr(y_true, y_pred)
    return r_score**2


def generate_scatter_plot(y_true, y_pred, label: str, min_val:int, max_val: int):
    fig, ax = plt.subplots()
    ax.scatter(x=y_pred, y=y_true, s=10, c='b')
    ax.grid(True)
    ax.plot([min_val, max_val], [min_val, max_val], c="gray", linestyle="--")

    ax.set_xlabel(f"Predicted {label}")
    ax.set_ylabel(f"Ground truth {label}")
    # fit a line, use linear regression
    a, b = np.polyfit(x=y_pred, y=y_true, deg=1)
    ax.plot([min_val, max_val], [a*min_val + b, a*max_val + b], c="r")

    mse = mean_squared_error(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)
    ax.set_title(f"r2: {r2:.3f}, mse: {mse:.3f}")

    return fig



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        correct_label=args.correct_label)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    total_train_batch_size = args.accum_freq * args.batch_size * args.world_size

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    if args.multimodal_type == 'oct3d_paired_faf_cls' or args.multimodal_type == 'oct3d_paired_ir_cls' or args.multimodal_type == 'oct3d_paired_faf_ir_cls':
        all_labels = []
        all_logits = []
        pearsonr_m = [AverageMeter() for _ in range(args.num_classes)]
        PearsonR_m = [AverageMeter() for _ in range(args.num_classes)]
        R2_m = [AverageMeter() for _ in range(args.num_classes)]
        r2_m = [AverageMeter() for _ in range(args.num_classes)]
        avg_mse_m = [AverageMeter() for _ in range(args.num_classes)]
        avg_mae_m = [AverageMeter() for _ in range(args.num_classes)]

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if args.multimodal_type == 'default':
            images, texts = batch
        else:

            images, texts_ir, texts_f2_faf = batch[0]['oct'], batch[0]['ir'], batch[0]['f2_faf']
            label = None
            img_name, modalities, h = batch[1]
            h, true_idx = h

            if args.multimodal_type == 'oct_ir':
                images, texts = images, texts_ir
            elif args.multimodal_type == 'oct_faf_only':
                assert sum(modalities[2]) == len(modalities[2]), 'Only f2_faf is allowed in this setting'
                images, texts = images, texts_f2_faf
            elif args.multimodal_type == 'oct_f2_faf_inplace':
                # FIXME: Neet to further develop
                print('modalities:', modalities)
                pass
            elif args.multimodal_type == 'oct_f2_faf_and_ir':
                # FIXME: Need to further develop
                pass
            elif args.multimodal_type == 'oct3d_paired_faf_cls':
                label = batch[0]['label']
                print(label, label.shape)
                texts = texts_f2_faf
            elif args.multimodal_type == 'oct3d_paired_ir_cls':
                label = batch[0]['label']
                print(label, label.shape)
                texts = texts_ir
            elif args.multimodal_type == 'oct3d_paired_faf_ir_cls':
                label = batch[0]['label']
                print(label, label.shape)
                texts_ir = texts_ir
                texts_faf = texts_f2_faf

        if not args.skip_scheduler:
            scheduler(step)


        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts_ir = texts_ir.to(device=device, non_blocking=True)
        texts_faf = texts_faf.to(device=device, non_blocking=True)
        label_cpu = label.float()
        label = label.float().to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if (args.multimodal_type == 'oct3d_paired_faf_cls' or args.multimodal_type == 'oct3d_paired_ir_cls' or args.multimodal_type == 'oct3d_paired_faf_ir_cls') and args.cls_dataset:
            if args.multimodal_type == 'oct3d_paited_faf_ir_cls':
                assert args.enable_3mod_training, '3 modality training is not enabled'
            with autocast():
                logits, logit_scale, logit_scale1, logit_scale2 = model(images, texts_ir, texts_faf, single_modality=args.single_modality)

                mse_loss = torch.nn.MSELoss(reduction='none')
                mae_loss = torch.nn.L1Loss(reduction='none')
                col_wise_loss_mse = mse_loss(logits, label).mean(dim=0)
                col_wise_loss_mae = mae_loss(logits, label).mean(dim=0)
                num_objectives = len(col_wise_loss_mse)
                weight_list = [0.1] + [1] * (num_objectives - 1)

                all_weights = sum(weight_list)
                # calculate total loss for both mse and mae
                total_loss = sum([weight_list[i] * col_wise_loss_mse[i] + weight_list[i] * col_wise_loss_mae[i] for i in range(num_objectives)]) / (2 * all_weights)

            backward(total_loss, scaler)
            all_labels.append(label_cpu)
            all_logits.append(logits.detach().cpu())

        else:
            if args.accum_freq == 1:
                with autocast():

                    image_features, text_features, logit_scale = model(images, texts)
                    total_loss = loss(image_features, text_features, logit_scale)

                backward(total_loss, scaler)
            else:
                # First, cache the features without any gradient tracking.
                with torch.no_grad():
                    with autocast():

                        chunk_image_features, chunk_text_features, _ = model(images, texts)
                    accum_image_features.append(chunk_image_features)
                    accum_text_features.append(chunk_text_features)

                    accum_images.append(images)
                    accum_texts.append(texts)

                # If (i + 1) % accum_freq is not zero, move on to the next batch.
                if ((i + 1) % args.accum_freq) > 0:
                    # FIXME this makes data time logging unreliable when accumulating
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                optimizer.zero_grad()

                # print if master
                if is_master(args):
                    print('i_accum:', i_accum, 'accum_freq:', args.accum_freq, i)

                for j in range(args.accum_freq):
                    images = accum_images[j]
                    texts = accum_texts[j]
                    with autocast():

                        chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
                        image_features = torch.cat(
                            accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                        text_features = torch.cat(
                            accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
                        total_loss = loss(image_features, text_features, logit_scale)
                    backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size * args.accum_freq
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {total_train_batch_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": total_train_batch_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                if args.fold != -1:
                    name = f"train_fold_{args.fold}/" + name
                else:
                    name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


    # end for

    if is_master(args) and (args.multimodal_type == 'oct3d_paired_faf_cls' or args.multimodal_type == 'oct3d_paired_ir_cls' or args.multimodal_type == 'oct3d_paired_faf_ir_cls') and args.cls_dataset:
        if args.multimodal_type == 'oct3d_paired_faf_ir_cls':
            assert args.enable_3mod_training, '3 modality training is not enabled'
        # calculate pearson correlation

        label_cpu = torch.cat(all_labels)
        logits = torch.cat(all_logits)
        print('label_cpu:', label_cpu.shape, 'logits:', logits.shape)
        for j in range(label.shape[1]):
            label_single = label_cpu[:, j]
            logits_single = logits[:, j].detach().cpu()

            PearsonR = np.corrcoef(label_single, logits_single)[0, 1]
            R2 = np.square(PearsonR)

            pearsonr_small = pearsonr(label_single, logits_single)[0]
            r2 = compute_r2(label_single, logits_single)

            mse = torch.mean((label_single - logits_single) ** 2)
            mae = torch.mean(torch.abs(label_single - logits_single))


            PearsonR_m[j].update(PearsonR)
            pearsonr_m[j].update(pearsonr_small)
            R2_m[j].update(R2)
            r2_m[j].update(r2)
            avg_mae_m[j].update(mae)
            avg_mse_m[j].update(mse)

            log_data[f'pearsonr_{j}'] = pearsonr_small
            log_data[f'r2_{j}'] = r2
            log_data[f'mse_{j}'] = mse.item()
            log_data[f'mae_{j}'] = mae.item()
            log_data[f'PearsonR_{j}'] = PearsonR
            log_data[f'R2_{j}'] = R2

            logging.info(f'pearsonr_{j}: {pearsonr_small}, r2_{j}: {r2}, mse_{j}: {mse.item()}, mae_{j}: {mae.item()}, PearsonR_{j}: {PearsonR}, R2_{j}: {R2}')
            print(log_data)
            for name, val in log_data.items():
                if args.fold != -1:
                    name = f"train_fold_{args.fold}/" + name
                else:
                    name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})




def evaluate(model, data, epoch, args, tb_writer=None, setting='val', dinfo_idx=0, return_prediction=False):
    assert setting in ['val', 'test', 'independent_test']
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    fold = args.fold
    save_intermediate_results_folder_name = f'log_{setting}/'
    if setting != 'val':
        save_intermediate_results_folder_name = save_intermediate_results_folder_name + f'{setting}_dataset_{dinfo_idx}/fold_{fold}/epoch_{epoch-1}/'
    else:
        save_intermediate_results_folder_name = save_intermediate_results_folder_name + f'{setting}_dataset_0/fold_{fold}/epoch_{epoch-1}/'
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    # print('zero_shot_metrics:', zero_shot_metrics)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    def get_corrected_label(text_features, t=10**(-8)):
        text_features_cpu = text_features.detach().cpu()
        text_exp_i = text_features_cpu.unsqueeze(1)
        text_exp_j = text_features_cpu.unsqueeze(0)
        dist_matrix = torch.sqrt(((text_exp_i - text_exp_j) ** 2).sum(dim=-1))
        L = (dist_matrix <= t).int().to(text_features.dtype)
        return L.to(text_features.device)

    if args.correct_label:
        criterion = torch.nn.BCEWithLogitsLoss()


    if (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        if isinstance(data, dict):
            datainfo = data[setting]
            dataloader = datainfo.dataloader
        else:
            dataloader = data.dataloader

        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME: this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0

        all_labels = []
        all_logits = []
        all_true_idx = []
        all_original_labels = []
        all_original_logits = []
        preset_label_mean = dataloader.dataset.preset_label_mean
        preset_label_std = dataloader.dataset.preset_label_std
        if preset_label_mean is not None and preset_label_std is not None:
            label_mean = torch.tensor(preset_label_mean).float()
            label_std = torch.tensor(preset_label_std).float()
        else:
            label_mean = dataloader.dataset.label_mean
            label_mean = torch.tensor(label_mean).float()
            label_std = dataloader.dataset.label_std
            label_std = torch.tensor(label_std).float()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                if args.multimodal_type == 'default':
                    if args.return_metainfo:
                        images, texts, labels = batch
                        all_labels.append(labels)
                    else:
                        images, texts = batch
                else:

                    images, texts_ir, texts_f2_faf = batch[0]['oct'], batch[0]['ir'], batch[0]['f2_faf']
                    img_names, modalities, h = batch[1]
                    h, true_idx = h
                    all_true_idx.append(true_idx)

                    if args.multimodal_type == 'oct_ir':
                        images, texts = images, texts_ir
                    elif args.multimodal_type == 'oct_faf_only':
                        images, texts = images, texts_f2_faf
                        assert sum(modalities[2]) == len(modalities[2]), 'Only f2_faf is allowed in this setting'

                    elif args.multimodal_type == 'oct3d_paired_faf_cls':
                        label = batch[0]['label']
                        texts = texts_f2_faf
                        label_cpu = label.float()
                        label = label.to(device=device, non_blocking=True).float()
                    elif args.multimodal_type == 'oct3d_paired_ir_cls':
                        label = batch[0]['label']
                        texts = texts_ir
                        label_cpu = label.float()
                        label = label.to(device=device, non_blocking=True).float()
                    elif args.multimodal_type == 'oct3d_paired_faf_ir_cls':
                        label = batch[0]['label']
                        texts_ir = texts_ir
                        texts_faf = texts_f2_faf
                        label_cpu = label.float()
                        label = label.to(device=device, non_blocking=True).float()

                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts_ir = texts_ir.to(device=device, non_blocking=True)
                texts_faf = texts_faf.to(device=device, non_blocking=True)


                with autocast():
                    # image_features, text_features, logit_scale = model(images, texts, img_coords, img_pad_mask)                    if args.multimodal_type == 'oct3d_paired_faf_cls' or args.multimodal_type == 'oct3d_paired_ir_cls' or args.multimodal_type == 'oct3d_paired_faf_ir_cls':
                        batch_size = images.shape[0]

                        logits, logit_scale, logit_scale1, logit_scale2 = model(images, texts_ir, texts_faf, single_modality=args.single_modality)


                        mse_loss = torch.nn.MSELoss(reduction='none')
                        mae_loss = torch.nn.L1Loss(reduction='none')
                        col_wise_loss_mse = mse_loss(logits, label).mean(dim=0)
                        col_wise_loss_mae = mae_loss(logits, label).mean(dim=0)

                        num_objectives = len(col_wise_loss_mse)
                        weight_list = [0.1] + [1] * (num_objectives - 1)
                        all_weights = sum(weight_list)
                        # calculate total loss for both mse and mae
                        total_loss = sum([weight_list[i] * col_wise_loss_mse[i] + weight_list[i] * col_wise_loss_mae[i] for i in range(num_objectives)]) / (2 * all_weights)

                        all_labels.append(label_cpu)
                        all_logits.append(logits.detach().cpu())
                        original_label = label_cpu * label_std + label_mean
                        original_logits = logits.detach().cpu() * label_std + label_mean
                        print('original_label:', original_label, 'original_logits:', original_logits, original_label.dtype, original_logits.dtype)
                        print('label_mean:', label_mean, 'label_std:', label_std, label_mean.dtype, label_std.dtype)

                        all_original_labels.append(original_label)
                        all_original_logits.append(original_logits)

                    else:
                        image_features, text_features, logit_scale = model(images, texts)

                        # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                        # however, system RAM is easily exceeded and compute time becomes problematic
                        all_image_features.append(image_features.cpu())
                        all_text_features.append(text_features.cpu())
                        logit_scale = logit_scale.mean()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()
                        all_logits_scale.append(logit_scale.cpu())
                        all_logits_per_image.append(logits_per_image.cpu())
                        all_logits_per_text.append(logits_per_text.cpu())

                        batch_size = images.shape[0]
                        if args.correct_label:
                            targets = get_corrected_label(text_features)
                            total_loss = (
                                criterion(logits_per_image, targets) +
                                criterion(logits_per_text, targets)
                            ) / 2
                        else:
                            targets = torch.arange(batch_size, device=device).long()
                            total_loss = (
                                F.cross_entropy(logits_per_image, targets) +
                                F.cross_entropy(logits_per_text, targets)
                            ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval {setting}-{dinfo_idx} Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")


            if args.multimodal_type == 'oct3d_paired_faf_cls' or args.multimodal_type == 'oct3d_paired_ir_cls' or args.multimodal_type == 'oct3d_paired_faf_ir_cls':
                # calculate pearson correlation

                label_cpu = torch.cat(all_labels)
                logits = torch.cat(all_logits)
                print('label_cpu:', label_cpu.shape, 'logits:', logits.shape)
                for j in range(label.shape[1]):
                    label_single = label_cpu[:, j]
                    logits_single = logits[:, j].detach().cpu()

                    PearsonR = np.corrcoef(label_single, logits_single)[0, 1]
                    R2 = np.square(PearsonR)

                    pearsonr_small = pearsonr(label_single, logits_single)[0]
                    r2 = compute_r2(label_single, logits_single)

                    mse = torch.mean((label_single - logits_single) ** 2)
                    mae = torch.mean(torch.abs(label_single - logits_single))


                    metrics[f'pearsonr_{j}'] = pearsonr_small
                    metrics[f'r2_{j}'] = r2
                    metrics[f'mse_{j}'] = mse.item()
                    metrics[f'mae_{j}'] = mae.item()
                    metrics[f'PearsonR_{j}'] = PearsonR
                    metrics[f'R2_{j}'] = R2

                    logging.info(f'{setting}-{dinfo_idx} pearsonr_{j}: {pearsonr_small}, r2_{j}: {r2}, mse_{j}: {mse.item()}, mae_{j}: {mae.item()}, PearsonR_{j}: {PearsonR}, R2_{j}: {R2}')
                    val_metrics = metrics

            else:
                val_metrics = get_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                )

            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics,  "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval {setting}-{dinfo_idx} Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if args.fold != -1:
                if setting == 'val':
                    name = f"val_fold_{args.fold}/" + name
                elif setting == 'test':
                    name = f"test_dataset_{dinfo_idx}_fold_{args.fold}/" + name
                elif setting == 'independent_test':
                    name = f"independent_test_dataset_{dinfo_idx}_fold_{args.fold}/" + name

            if tb_writer is not None:
                tb_writer.add_scalar(f"{setting}-{dinfo_idx}/{name}", val, epoch)

        result_name = f"results-{setting}-{dinfo_idx}.jsonl" if args.fold == -1 else f"results-{setting}-{dinfo_idx}_fold_{args.fold}.jsonl"
        with open(os.path.join(args.checkpoint_path, result_name), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")


    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if setting == 'val':
            val_name = "val" if args.fold == -1 else f"val_fold_{args.fold}"
        elif setting == 'test' or setting == 'independent_test':
            val_name = f"{setting}_{dinfo_idx}" if args.fold == -1 else f"{setting}_{dinfo_idx}_fold_{args.fold}"

        for name, val in metrics.items():
            wandb.log({f"{val_name}/{name}": val, 'epoch': epoch})

        original_labels = torch.cat(all_original_labels)
        original_logits = torch.cat(all_original_logits)
        original_true_idx = torch.cat(all_true_idx)

        plot_name = ['GAArea', 'BCVABASE', 'GAGrowth', 'BCVARATE', 'BCVACHG72']
        min_val_list = [0, 40, 0, -50, -50]
        max_val_list = [20, 80, 5, 20, 20]
        if args.cls_dataset_type == 'GAGrowth' or args.cls_dataset_type == 'GAGrowth_eyenotate' or args.cls_dataset_type == 'GAGrowth_OCTCorr':
            plot_name = ['GAArea', 'GAGrowth']
            min_val_list = [0, 0]
            max_val_list = [20, 5]

        # Save results and plot
        for i in range(original_labels.shape[1]):
            poly_coef = np.polyfit(original_labels[:, i].numpy(), original_logits[:, i].numpy(), 1)
            save_data = {
                'label': str(plot_name[i]),
                'min_val': int(min_val_list[i]),
                'max_val': int(max_val_list[i]),
                'epoch': epoch,
                'actual_epoch': epoch-1,
                'fold': args.fold,
                'cls_dataset_type': args.cls_dataset_type,
                'poly_coef': poly_coef.tolist(),
                'true_idx': original_true_idx.numpy().tolist(),
                'y_true': original_labels[:, i].numpy().tolist(),
                'y_pred': original_logits[:, i].numpy().tolist(),
            }
            if not os.path.exists(os.path.join(args.checkpoint_path, save_intermediate_results_folder_name)):
                os.makedirs(os.path.join(args.checkpoint_path, save_intermediate_results_folder_name))

            # save the data
            with open(os.path.join(args.checkpoint_path, save_intermediate_results_folder_name, f'{plot_name[i]}.pkl'), 'wb') as f:
                pkl.dump(save_data, f)
            # save the data into json
            with open(os.path.join(args.checkpoint_path, save_intermediate_results_folder_name, f'{plot_name[i]}.json'), 'w') as f:
                json.dump(save_data, f, indent=2)

            # Scatter plots predictions vs ground truth
            fig = generate_scatter_plot(
                y_true=original_labels[:, i],  # y_true=
                y_pred=original_logits[:, i],  # y_pred=
                label=plot_name[i],
                min_val=min_val_list[i],
                max_val=max_val_list[i],
            )
            wandb.log(
                {
                    f"{val_name}/scatter_{plot_name[i]}_{epoch}_r2_{metrics['r2_'+str(i)]:.4f}": wandb.Image(fig)
                }
            )


    dicted_results = {
        'original_labels': original_labels.numpy(),
        'original_logits': original_logits.numpy(),
        'original_true_idx': original_true_idx.numpy(),
    }
    # print(original_labels.shape, original_logits.shape, original_true_idx.shape)


    if return_prediction:
        return metrics, dicted_results

    return metrics

def plot_results(args, original_labels, original_logits):
    plot_name = ['GAArea', 'BCVABASE', 'GAGrowth', 'BCVARATE', 'BCVACHG72']
    min_val_list = [0, 40, 0, -50, -50]
    max_val_list = [20, 80, 5, 20, 20]
    if args.cls_dataset_type == 'GAGrowth' or args.cls_dataset_type == 'GAGrowth_eyenotate' or args.cls_dataset_type == 'GAGrowth_OCTCorr':
        plot_name = ['GAArea', 'GAGrowth']
        min_val_list = [0, 0]
        max_val_list = [20, 5]

    figs_list = []
    if isinstance(original_labels, torch.Tensor):
        original_labels = original_labels.numpy()
    if isinstance(original_logits, torch.Tensor):
        original_logits = original_logits.numpy()
    # Save results and plot
    for i in range(original_labels.shape[1]):
        poly_coef = np.polyfit(original_labels[:, i], original_logits[:, i], 1)

        # Scatter plots predictions vs ground truth
        fig = generate_scatter_plot(
            y_true=original_labels[:, i],
            y_pred=original_logits[:, i],
            label=plot_name[i],
            min_val=min_val_list[i],
            max_val=max_val_list[i],
        )
        figs_list.append(fig)
    return figs_list





def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def make_correct_labels(logits, labels):
    '''
    Merge the same reports as one class
    '''
    reduced_labels = list(set(labels))
    label2idx = {labels[i]: i for i in range(len(labels))}
    reduced_logits = [logits[:, label2idx[reduced_labels[i]]].reshape(-1, 1) for i in range(len(reduced_labels))]
    reduced_logits = np.concatenate(reduced_logits, axis=1)
    label_ints = np.asarray([reduced_labels.index(l) for l in labels])

    return torch.from_numpy(reduced_logits), torch.from_numpy(label_ints)

def get_corrected_metrics(image_features, text_features, logit_scale, labels):
    labels_np = np.asarray(labels)

    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    # notably, the labels array should be symmetric
    for name, logit in logits.items():
        if name == 'image_to_text':
            # In this case, one image can only be mapped to one text, but the text might
            # appear several times, caused by other image associated with the same text,
            # therefore we need to reduce the redundant logits
            corrected_logits, corrected_labels = make_correct_labels(logit.numpy(), labels_np)
            ranking = torch.argsort(corrected_logits, descending=True)
            preds = torch.where(ranking == corrected_labels.view(-1, 1))[1]
            preds = preds.detach().cpu().numpy()
            metrics[f"corrected_{name}_mean_rank"] = preds.mean() + 1
            metrics[f"corrected_{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                metrics[f"corrected_{name}_R@{k}"] = np.mean(preds < k)
        elif name == 'text_to_image':
            # In this case, one text can have many counterpart images, therefore, it's multi-label
            # classification problem, in this setting, only recall can be computed
            targets = np.vstack([(labels_np == labels_np[i]).reshape(1, -1) for i in range(len(labels_np))]).astype(int)
            preds = (torch.sigmoid(logits_per_text) >= 0.5).int().numpy()
            micro_recall = recall_score(targets, preds, average='micro')
            macro_recall = recall_score(targets, preds, average='macro')
            metrics[f"corrected_{name}_micro_recall"] = micro_recall
            metrics[f"corrected_{name}_macro_recall"] = macro_recall
    return metrics
