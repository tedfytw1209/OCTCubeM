
# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import math
from typing import Iterable
import pickle as pkl

import custom_util.lr_sched as lr_sched
import custom_util.misc as misc
import torch
from iopath.common.file_io import g_pathmgr as pathmgr
import torch.nn as nn
import torch.nn.functional as F



def train_one_epoch_joint(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    data_loader_2d,
    dataset_2d_all_image_dict,
    mask_ratio_2d,
    log_writer=None,
    args=None,
    fp32=False,
    fp16=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio_2d", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    metric_logger.add_meter(
        "loss_2d", misc.SmoothedValue(window_size=1)
    )
    metric_logger.add_meter(
        'loss_all', misc.SmoothedValue(window_size=1)
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    secondary_iter = iter(data_loader_2d)  # Create an iterator for the secondary data loader
    print('Initiate secondary data loader', f'len(data_loader_2d): {len(data_loader_2d)}', len(data_loader_2d)*args.batch_size_2d)

    for data_iter_step, (samples, data_info) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )
        img_names = data_info[0]
        data_dict = data_info[1]

        try:
            secondary_data = next(secondary_iter)
        except StopIteration:
            secondary_iter = iter(data_loader_2d)  # Restart the iterator
            secondary_data = next(secondary_iter)

        sample_2d = secondary_data[0]
        sample_2d_info = secondary_data[1]

        samples = samples.to(device, non_blocking=True)

        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)



        with torch.cuda.amp.autocast(enabled=not fp32, dtype=torch.float16 if fp16 else None):

            feat = model.module.forward_patch_embed(samples).detach()

            filled_mask_list = misc.get_mask(feat)
            filled_mask_tensor = torch.stack(filled_mask_list, dim=0).long()

            loss, _, _ = model(
                samples,
                mask_ratio=args.mask_ratio,
                frame_loss=True,
                pre_mask=filled_mask_tensor,
            )

            loss_2d, _, _ = model(
                sample_2d,
                mask_ratio=mask_ratio_2d,
            )
        loss, frame_loss = loss

        loss_value = loss.item()
        loss_2d_value = loss_2d.item()

        for j, vol in enumerate(frame_loss):
            cube_size = 3
            frame_name_list = [data_dict['frames'][nf][j] for nf in range(len(data_dict['frames']))]

            for k, frame_loss in enumerate(vol):
                # k ranges from 0 to 20
                # each frame_loss goes for 3 frames
                for fr in range(cube_size):
                    frame_name = frame_name_list[k * cube_size + fr]
                    dataset_2d_all_image_dict[frame_name]['mse_loss'] = frame_loss.item()
                    dataset_2d_all_image_dict[frame_name]['hardness'] = frame_loss.item()

            dataset_2d_all_image_dict[frame_name_list[-1]]['mse_loss'] = frame_loss.item()
            dataset_2d_all_image_dict[frame_name_list[-1]]['hardness'] = frame_loss.item()


        loss = loss + loss_2d
        loss_all_value = loss.item()


        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=args.mask_ratio)
        metric_logger.update(loss_2d=loss_2d_value)
        metric_logger.update(loss_all=loss_all_value)
        metric_logger.update(mask_ratio_2d=mask_ratio_2d)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
    fp32=False,
    joint=False,
    mask_ratio_2d=None,
    data_loader_2d=None,
    visible_frame_freq=20,
    all_image_dict=None,
    analysis=False,
    fp16=False,
    agg_data=False,
):

    img_save_dir = os.path.join(args.output_dir, f'val_images_{epoch}')
    os.makedirs(img_save_dir, exist_ok=True)

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    if joint and data_loader_2d is not None:
        secondary_iter = iter(data_loader_2d)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, img_names) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        print(img_names)
        samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        with torch.cuda.amp.autocast(enabled=not fp32, dtype=torch.float16 if fp16 else None):

            feat = model.module.forward_patch_embed(samples)
            if analysis:
                pass
                # print(feat.shape)
                # exit()
            loss, pred, mask = model(
                samples,
                mask_ratio=args.mask_ratio,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        torch.cuda.synchronize()

        # log the volume reconstruction results
        if (data_iter_step)  % (print_freq * visible_frame_freq) == 0:
            if misc.is_main_process():
                print('joint')
                if joint:
                    print('goin', img_names)
                    img_names = img_names[0]
                vars_ = {'reconstruct_imgs': pred,
                                    'samples': samples,
                                    'mask': mask,
                                    'img_names': img_names,
                                    'patch_embed': feat,}

                misc.get_visible_images(vars_, model, img_save_dir)
                if not agg_data:
                    misc.get_patch_embed_images(vars_, model, img_save_dir, img_type='3D')

                if joint and data_loader_2d is not None:
                    secondary_data = next(secondary_iter)
                    sample_2d = secondary_data[0]
                    sample_2d_info = secondary_data[1]
                    img_names_2d = sample_2d_info[-1]

                    pre_mask = torch.zeros_like(sample_2d)
                    for i in range(len(sample_2d)):
                        sample = sample_2d[i, 0]

                        sample_2d[i, 0], pre_mask[i, 0] = misc.find_and_convert_large_white_region(sample)

                    with torch.cuda.amp.autocast(enabled=not fp32, dtype=torch.float16 if fp16 else None):
                        loss_2d, pred_2d, mask_2d = model(
                            sample_2d,
                            mask_ratio=mask_ratio_2d,
                        )
                    feat = model.module.forward_patch_embed(sample_2d.to(device, non_blocking=True))
                    vars_2d_ = {'reconstruct_imgs': pred_2d,
                                    'samples': sample_2d,
                                    'mask': mask_2d,
                                    'img_names': img_names_2d,
                                    'patch_embed': feat,
                                    'pre_mask': pre_mask,}
                    misc.get_visible_images_2d(vars_2d_, model, img_save_dir)
                    if not agg_data:
                        misc.get_patch_embed_images(vars_2d_, model, img_save_dir, img_type='2D', use_pre_mask=True)

                        sample_2d_256 = F.interpolate(sample_2d.squeeze(1), size=(256, 256), mode='bilinear', align_corners=False).unsqueeze(1)
                        pre_mask_256 = F.interpolate(pre_mask.squeeze(1).float(), size=(256, 256), mode='bilinear', align_corners=False).unsqueeze(1)
                        pre_mask_256 = torch.round(pre_mask_256).bool()

                        feat_256 = model.module.forward_patch_embed(sample_2d_256.to(device, non_blocking=True))
                        vars_2d_256 = {'reconstruct_imgs': pred_2d,
                                        'samples': sample_2d_256,
                                        'mask': mask_2d,
                                        'img_names': img_names_2d,
                                        'patch_embed': feat_256,
                                        'pre_mask': pre_mask_256,}
                        misc.get_patch_embed_images(vars_2d_256, model, img_save_dir, img_type='2D', use_pre_mask=True)
                    print(f'Save 2d images for iter {print_freq * visible_frame_freq} frames')

        metric_logger.update(loss=loss_value)
        metric_logger.update(mask_ratio=args.mask_ratio)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("val_loss", loss_value_reduce, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

