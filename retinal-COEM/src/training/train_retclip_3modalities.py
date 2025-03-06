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

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from sklearn.metrics import recall_score
from open_clip import ClipLoss, get_cast_dtype, convert_modalities_idx_to_flag, ThreeModalityClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


def stack_weight_modalities(weight_modalities):
    accum_steps = len(weight_modalities)
    num_modalities = len(weight_modalities[0])
    new_weight_modalities = []
    for i in range(num_modalities):
        single_modality_weight = torch.cat([weight_modalities[j][i] for j in range(accum_steps)], dim=0)
        # print('single_modality_weight:', single_modality_weight.shape, single_modality_weight)
        new_weight_modalities.append(single_modality_weight)
    return new_weight_modalities

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
    loss = ThreeModalityClipLoss(
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
        accum_images, accum_texts_ir, accum_texts_f2_faf, accum_image_features, accum_text_ir_features, accum_text_faf_features = [], [], [], [], [], []
        accum_weight_modalities = []
    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if args.multimodal_type == 'default':
            images, texts = batch
        else:

            images, texts_ir, texts_f2_faf = batch[0]['oct'], batch[0]['ir'], batch[0]['f2_faf']
            img_name, dataset_idx, modalities, h = batch[1]

            oct_modalities = convert_modalities_idx_to_flag(modalities[0])
            ir_modalities = convert_modalities_idx_to_flag(modalities[1])
            f2_faf_modalities = convert_modalities_idx_to_flag(modalities[2])

            weight_modalities = [oct_modalities, ir_modalities, f2_faf_modalities]

            if args.multimodal_type == 'oct_ir':
                images, texts = images, texts_ir
            elif args.multimodal_type == 'oct_faf_only':
                assert sum(modalities[2]) == len(modalities[2]), 'Only f2_faf is allowed in this setting'
                images, texts = images, texts_f2_faf
            elif args.multimodal_type == 'oct_f2_faf_inplace':

                pass
            elif args.multimodal_type == 'oct_faf_ir':
                images, texts_faf, texts_ir = images, texts_f2_faf, texts_ir

        if not args.skip_scheduler:
            scheduler(step)


        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts_ir = texts_ir.to(device=device, non_blocking=True)
        texts_f2_faf = texts_f2_faf.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():

                image_features, text_ir_features, text_faf_features, logit_scale, logit_scale1, logit_scale2 = model(images, texts_ir, texts_f2_faf)

                total_loss = loss(image_features, text_ir_features, text_faf_features, logit_scale, logit_scale1, logit_scale2, t_weight1=weight_modalities[1], t_weight2=weight_modalities[2])


            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():

                    chunk_image_features, chunk_text_ir_features, chunk_text_faf_features, _, _, _ = model(images, texts_ir, texts_f2_faf)

                accum_image_features.append(chunk_image_features)
                accum_text_ir_features.append(chunk_text_ir_features)
                accum_text_faf_features.append(chunk_text_faf_features)

                accum_images.append(images)
                accum_texts_ir.append(texts_ir)
                accum_texts_f2_faf.append(texts_f2_faf)
                accum_weight_modalities.append(weight_modalities)

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

            concat_accum_weight_modalities = stack_weight_modalities(accum_weight_modalities)
            # send to device
            concat_accum_weight_modalities = [item.to(device=device, dtype=cast_dtype, non_blocking=True) for item in concat_accum_weight_modalities]


            for j in range(args.accum_freq):
                images = accum_images[j]
                texts_ir = accum_texts_ir[j]
                texts_f2_faf = accum_texts_f2_faf[j]

                with autocast():

                    chunk_image_features, chunk_text_ir_features, chunk_text_faf_features, logit_scale, logit_scale1, logit_scale2 = model(images, texts_ir, texts_f2_faf)
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                    text_ir_features = torch.cat(
                        accum_text_ir_features[:j] + [chunk_text_ir_features] + accum_text_ir_features[j + 1:])
                    text_faf_features = torch.cat(
                        accum_text_faf_features[:j] + [chunk_text_faf_features] + accum_text_faf_features[j + 1:])


                    total_loss = loss(image_features, text_ir_features, text_faf_features, logit_scale, logit_scale1, logit_scale2, t_weight1=concat_accum_weight_modalities[1], t_weight2=concat_accum_weight_modalities[2])

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
            # accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
            accum_images, accum_texts_ir, accum_texts_f2_faf, accum_image_features, accum_text_ir_features, accum_text_faf_features = [], [], [], [], [], []
            accum_weight_modalities = []
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


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

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

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME: this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_ir_features, all_text_faf_features = [], [], []
        all_logits_scale = []
        all_logits_scale1 = []
        all_logits_scale2 = []

        all_logits_per_image2text_ir = []
        all_logits_per_text_ir2image = []
        all_logits_per_image2text_faf = []
        all_logits_per_text_faf2image = []
        all_logits_per_text_ir2text_faf = []
        all_logits_per_text_faf2text_ir = []

        all_weight_modalities = []



        all_labels = []
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
                    img_names, dataset_idx, modalities, h = batch[1]

                    oct_modalities = convert_modalities_idx_to_flag(modalities[0])
                    ir_modalities = convert_modalities_idx_to_flag(modalities[1])
                    f2_faf_modalities = convert_modalities_idx_to_flag(modalities[2])
                    weight_modalities = [oct_modalities, ir_modalities, f2_faf_modalities]

                    if args.multimodal_type == 'oct_ir':
                        images, texts = images, texts_ir
                    elif args.multimodal_type == 'oct_faf_only':
                        images, texts = images, texts_f2_faf
                        assert sum(modalities[2]) == len(modalities[2]), 'Only f2_faf is allowed in this setting'

                    elif args.multimodal_type == 'oct_f2_faf_inplace':
                        # FIXME: Neet to further develop
                        print('modalities:', modalities)
                        pass
                    elif args.multimodal_type == 'oct_faf_ir':
                        images, texts_faf, texts_ir = images, texts_f2_faf, texts_ir


                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)

                texts_ir = texts_ir.to(device=device, non_blocking=True)
                texts_f2_faf = texts_f2_faf.to(device=device, non_blocking=True)



                with autocast():

                    image_features, text_ir_features, text_faf_features, logit_scale, logit_scale1, logit_scale2 = model(images, texts_ir, texts_f2_faf)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_ir_features.append(text_ir_features.cpu())
                    all_text_faf_features.append(text_faf_features.cpu())

                    logit_scale = logit_scale.mean()
                    logit_scale1 = logit_scale1.mean()
                    logit_scale2 = logit_scale2.mean()

                    logits_per_image_text_ir = logit_scale * image_features @ text_ir_features.T
                    logits_per_text_ir_image = logits_per_image_text_ir.T

                    logits_per_image_text_faf = logit_scale1 * image_features @ text_faf_features.T
                    logits_per_text_faf_image = logits_per_image_text_faf.T

                    logits_per_text_ir_text_faf = logit_scale2 * text_ir_features @ text_faf_features.T
                    logits_per_text_faf_text_ir = logits_per_text_ir_text_faf.T


                    all_logits_scale.append(logit_scale.cpu())
                    all_logits_scale1.append(logit_scale1.cpu())
                    all_logits_scale2.append(logit_scale2.cpu())

                    all_logits_per_image2text_ir.append(logits_per_image_text_ir.cpu())
                    all_logits_per_text_ir2image.append(logits_per_text_ir_image.cpu())
                    all_logits_per_image2text_faf.append(logits_per_image_text_faf.cpu())
                    all_logits_per_text_faf2image.append(logits_per_text_faf_image.cpu())
                    all_logits_per_text_ir2text_faf.append(logits_per_text_ir_text_faf.cpu())
                    all_logits_per_text_faf2text_ir.append(logits_per_text_faf_text_ir.cpu())

                    all_weight_modalities.append(weight_modalities)



                    batch_size = images.shape[0]
                    if args.correct_label:
                        targets = get_corrected_label(text_ir_features)

                    else:
                        targets = torch.arange(batch_size, device=device).long()

                    used_t_weight1 = weight_modalities[1]
                    used_t_weight2 = weight_modalities[2]

                    used_t_weight1 = used_t_weight1.to(device=device, dtype=cast_dtype, non_blocking=True)
                    used_t_weight2 = used_t_weight2.to(device=device, dtype=cast_dtype, non_blocking=True)

                    # Image-Text1 Loss
                    loss_it1 = F.cross_entropy(logits_per_image_text_ir, targets, reduction='none')
                    loss_t1i = F.cross_entropy(logits_per_text_ir_image, targets, reduction='none')
                    # Apply masks
                    loss_it1 = loss_it1 * used_t_weight1
                    loss_t1i = loss_t1i * used_t_weight1

                    # Average over valid samples
                    if used_t_weight1.sum() == 0:
                        loss_it1 = torch.tensor(0.0, device=device)
                        loss_t1i = torch.tensor(0.0, device=device)
                    else:
                        loss_it1 = loss_it1.sum() / used_t_weight1.sum()
                        loss_t1i = loss_t1i.sum() / used_t_weight1.sum()

                    # Image-Text2 Loss
                    loss_it2 = F.cross_entropy(logits_per_image_text_faf, targets, reduction='none')
                    loss_t2i = F.cross_entropy(logits_per_text_faf_image, targets, reduction='none')
                    # Apply masks
                    loss_it2 = loss_it2 * used_t_weight2
                    loss_t2i = loss_t2i * used_t_weight2

                    # Average over valid samples
                    if used_t_weight2.sum() == 0:
                        loss_it2 = torch.tensor(0.0, device=device)
                        loss_t2i = torch.tensor(0.0, device=device)
                    else:
                        loss_it2 = loss_it2.sum() / used_t_weight2.sum()
                        loss_t2i = loss_t2i.sum() / used_t_weight2.sum()

                    # Text1-Text2 Loss
                    loss_t1t2 = F.cross_entropy(logits_per_text_ir_text_faf, targets, reduction='none')
                    loss_t2t1 = F.cross_entropy(logits_per_text_faf_text_ir, targets, reduction='none')
                    # Apply masks
                    loss_t1t2 = loss_t1t2 * used_t_weight1
                    loss_t2t1 = loss_t2t1 * used_t_weight1

                    # Average over valid samples
                    if used_t_weight1.sum() == 0:
                        loss_t1t2 = torch.tensor(0.0, device=device)
                        loss_t2t1 = torch.tensor(0.0, device=device)
                    else:
                        loss_t1t2 = loss_t1t2.sum() / used_t_weight1.sum()
                        loss_t2t1 = loss_t2t1.sum() / used_t_weight1.sum()


                    # Total loss is the average of all valid losses
                    total_loss = (loss_it1 + loss_t1i + loss_it2 + loss_t2i + loss_t1t2 + loss_t2t1) / 6

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            concat_all_weights = stack_weight_modalities(all_weight_modalities)
            print('concat_all_weights:', concat_all_weights[1], concat_all_weights[2], sum(concat_all_weights[1].float()), sum(concat_all_weights[2].float()))

            val_metrics = get_metrics_3modalities(
                image_features=torch.cat(all_image_features),
                text1_features=torch.cat(all_text_ir_features),
                text2_features=torch.cat(all_text_faf_features),
                logit_scale=logit_scale.cpu(),
                logit_scale1=logit_scale1.cpu(),
                logit_scale2=logit_scale2.cpu(),
                t_weight1=concat_all_weights[1],
                t_weight2=concat_all_weights[2],
            )

            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics,  "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")
        if args.save_retrieval_results:
            # save all_image_features and all_text_features, and all_labels, if available
            # using pickle
            import pickle as pkl
            print('Saving retrieval results...')
            print(all_labels[0])
            print('all_labels:', len(all_labels))

            with open(os.path.join(args.checkpoint_path, f"retrieval_results_{epoch}.pkl"), "wb") as f:
                results = {
                    "image_features": torch.cat(all_image_features).numpy(),
                    "text_features": torch.cat(all_text_features).numpy(),
                    "labels": all_labels,
                    "logit_scale": all_logits_scale,
                    "logits_per_image": all_logits_per_image,
                    "logits_per_text": all_logits_per_text
                }
                print(all_logits_per_image)
                print(all_logits_per_text)
                pkl.dump(results, f)


    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics



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

def get_metrics_3modalities(image_features, text1_features, text2_features, logit_scale, logit_scale1, logit_scale2, t_weight1, t_weight2):
    metrics = {}
    logits_per_image_text1 = (logit_scale * image_features @ text1_features.t()).detach().cpu()
    logits_per_text1_image = logits_per_image_text1.t().detach().cpu()

    logits_per_image_text2 = (logit_scale1 * image_features @ text2_features.t()).detach().cpu()
    logits_per_text2_image = logits_per_image_text2.t().detach().cpu()

    logits_per_text1_text2 = (logit_scale2 * text1_features @ text2_features.t()).detach().cpu()
    logits_per_text2_text1 = logits_per_text1_text2.t().detach().cpu()

    logits = {"image_to_text1": logits_per_image_text1, "text1_to_image": logits_per_text1_image,
                "image_to_text2": logits_per_image_text2, "text2_to_image": logits_per_text2_image,
                "text1_to_text2": logits_per_text1_text2, "text2_to_text1": logits_per_text2_text1}
    t_weight1 = t_weight1.cpu()
    t_weight2 = t_weight2.cpu()
    t_weight_pair = t_weight1 * t_weight2

    ground_truth = torch.arange(len(text1_features)).view(-1, 1)

    for name, logit in logits.items():
        if 'image' in name:
            if 'text1' in name:
                t_weight = t_weight1
            elif 'text2' in name:
                t_weight = t_weight2
            else:
                raise ValueError('Invalid name')
        else:
            assert 'text1' in name and 'text2' in name, 'Invalid name'
            t_weight = t_weight_pair

        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        mean_pred_filter_masked = preds[t_weight > 0]
        print('Lens before:', len(preds), 'After:', len(mean_pred_filter_masked))
        metrics[f"{name}_mean_rank"] = mean_pred_filter_masked.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(mean_pred_filter_masked)) + 1
        for k in [1, 5, 10]:
            # metrics[f"{name}_R@{k}"] = np.mean(preds < k)
            metrics[f"{name}_R@{k}"] = np.mean(mean_pred_filter_masked < k)

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
