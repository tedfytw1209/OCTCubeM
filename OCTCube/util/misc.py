# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from iopath.common.file_io import g_pathmgr as pathmgr
import torch
import torch.distributed as dist
if torch.__version__ >= '2.0.0':
    from torch import inf
else:
    from torch._six import inf
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as tf

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if int(args.rank)==-1 and not args.dist_on_itp: ##!!TMP FIX
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return
    elif args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)

            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_add_dir="", mode='best'):
    #output_dir = Path(args.output_dir + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(args.output_dir, model_add_dir), exist_ok=True)
    #epoch_name = str(epoch)
    if loss_scaler is not None:
        # checkpoint_paths = [args.task+f'checkpoint-{epoch}.pth']
        if mode == 'best':
            checkpoint_paths = [os.path.join(args.output_dir, model_add_dir, 'checkpoint-best.pth')]
        elif mode == 'epoch':
            checkpoint_paths = [os.path.join(args.output_dir, model_add_dir, 'checkpoint-%s.pth' % epoch_name)]
        else:
            raise ValueError('Invalid mode')
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                #'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                #'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=os.path.join(args.output_dir, model_add_dir), tag="checkpoint-best", client_state=client_state)


def get_last_checkpoint(args, model_add_dir=""):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = os.path.join(args.output_dir, model_add_dir)
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        return os.path.join(d, name)

def load_model(args, model_without_ddp, optimizer=None, loss_scaler=None, only_model=False, model_add_dir=""):
    if args.resume == 'latest':
        args.resume = get_last_checkpoint(args, model_add_dir=model_add_dir)
    print('resume pth:', args.resume)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        interpolate_pos_embed(model_without_ddp, checkpoint['model'])
        interpolate_temporal_pos_embed(model_without_ddp, checkpoint['model'])
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if not only_model and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
    else:
        raise ValueError("Checkpoint '%s' not found" % args.resume)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def master_print(*args, **kwargs):
    if is_master_proc():
        print(*args, **kwargs)
    else:
        pass


def is_master_proc(multinode=False):
    """
    Determines if the current process is the master process.
    """
    if dist.is_initialized():
        if multinode:
            return dist.get_rank() % dist.get_world_size() == 0
        else:
            return dist.get_rank() % torch.cuda.device_count() == 0
    else:
        return True

def get_patient_dataset_cnt_dict(dataset):
    cnt_dict = {}
    print('len(dataset):', len(dataset))
    for i in range(len(dataset)):
        data_dict = dataset.visits_dict[i]
        patient_id = dataset.mapping_visit2patient[i]
        cls_id = data_dict['class_idx']
        if cls_id not in cnt_dict:
            cnt_dict[cls_id] = 1
        else:
            cnt_dict[cls_id] += 1
    return cnt_dict

def general_generate_sublists(cnt_dict, k=3, r=0.3, iterations=10, random_state=42, return_rest_lists=True, beta=1.0, fewshot=False):
    # Validate input
    if k < 1 or not (0 < r <= 1):
        raise ValueError("Invalid values for k or r. Ensure k >= 1 and 0 < r <= 1.")

    total_samples = sum(cnt_dict.values())
    num_classes = len(cnt_dict)
    target_list_size = int(np.ceil(total_samples * r))

    # Initialize numpy RandomState
    rng = np.random.RandomState(random_state)

    # Create a list of samples for each class
    classes = []
    start = 0
    for count in cnt_dict.values():
        classes.append(list(range(start, start + count)))
        start += count

    # Initialize selection counts
    selection_counts = [np.zeros(len(cls_list), dtype=int) for cls_list in classes]
    generated_lists = []
    rest_lists = []

    for _ in range(iterations):
        combined_list = []
        combined_cls_list = []
        # Calculate the minimum samples to pick from each class
        total_picked = 0
        for i, cls_list in enumerate(classes):
            if len(cls_list) < k:
                raise ValueError(f"Class {i} does not have enough samples to pick {k} items.")

            # Choose k elements randomly
            selected_indices = rng.choice(len(cls_list), size=k, replace=False)
            selection_counts[i][selected_indices] += 1
            combined_list.extend([cls_list[idx] for idx in selected_indices])
            total_picked += k

        # Pick additional elements if required, updating weights after each pick
        extra_picks = target_list_size - total_picked
        while extra_picks > 0:
            all_indices = []
            all_weights = []
            for i, cls_list in enumerate(classes):
                weights = np.exp(-beta * selection_counts[i])
                weights /= weights.sum()  # Normalize weights
                all_weights.extend(weights)
                all_indices.extend([(i, idx) for idx in range(len(cls_list))])

            if np.sum(all_weights) != 1:
                all_weights /= np.sum(all_weights)  # Ensure sum of probabilities is exactly 1

            additional_index = rng.choice(len(all_indices), p=all_weights)
            cls_idx, idx = all_indices[additional_index]
            selection_counts[cls_idx][idx] += 1
            combined_list.append(classes[cls_idx][idx])
            combined_cls_list.append(cls_idx)
            extra_picks -= 1
        print('combined_cls_list:', combined_cls_list, len(combined_cls_list), len(combined_list))
        # Calculate rest list
        rest_list = [x for cls_list in classes for x in cls_list if x not in combined_list]

        generated_lists.append(combined_list)
        rest_lists.append(rest_list)

    if return_rest_lists:
        # Return the list in tuple of two for each iteration
        return_list = []
        for i in range(iterations):
            if fewshot:
                return_list.append((np.array(rest_lists[i]), np.array(generated_lists[i])))
            else:
                return_list.append((np.array(generated_lists[i]), np.array(rest_lists[i])))
        return return_list
    else:
        return generated_lists



# for duke 14 effective sampling
def generate_sublists(full_list, N=15, k=2, iterations=10, random_state=42, return_rest_lists=True):
    if len(full_list) != 3 * N:
        raise ValueError("The list must be of length 3N.")

    # Initialize numpy RandomState
    rng = np.random.RandomState(random_state)

    # Split the list into three classes
    classes = [full_list[:N], full_list[N:2*N], full_list[2*N:]]

    # Initialize selection counts
    selection_counts = [np.zeros(N, dtype=int) for _ in range(3)]
    rest_lists = []
    generated_lists = []
    for _ in range(iterations):
        combined_list = []
        for i, cls in enumerate(classes):
            # Calculate weights inversely proportional to the selection count
            weights = 1.0 / (1.0 + selection_counts[i])
            weights /= weights.sum()

            # Choose k elements based on weights
            selected_indices = rng.choice(N, size=k, replace=False, p=weights)
            selection_counts[i][selected_indices] += 1  # Update counts

            # Add selected elements to the combined list
            combined_list.extend([cls[idx] for idx in selected_indices])
        rest_list = [full_list[i] for i in range(3*N) if i not in combined_list]

        generated_lists.append(combined_list)
        rest_lists.append(rest_list)
    if return_rest_lists:
        # return the list in tuple of two for each iteration
        return_list = []
        for i in range(iterations):
            return_list.append((np.array(rest_lists[i]), np.array(generated_lists[i])))
        return return_list
    else:
        return generated_lists

def hcms_generate_sublists(full_list, N1=14, N2=21, k=3, iterations=10, random_state=42, return_rest_lists=True):
    if len(full_list) != N1 + N2:
        raise ValueError("The list must match the sum of the two class sizes (N1 + N2).")

    # Initialize numpy RandomState
    rng = np.random.RandomState(random_state)

    # Split the list into two classes based on given N1 and N2
    classes = [full_list[:N1], full_list[N1:]]

    # Initialize selection counts
    selection_counts = [np.zeros(len(cls), dtype=int) for cls in classes]
    rest_lists = []
    generated_lists = []
    if isinstance(k, int):
        k_list = [k for _ in len(classes)]
    elif isinstance(k, list):
        assert len(k) == len(classes), "The number of k values must match the number of classes."
        k_list = k
    else:
        raise ValueError("Invalid value for k. Must be an integer or a list of integers.")
    for _ in range(iterations):
        combined_list = []
        for i, cls in enumerate(classes):
            used_k = k_list[i]
            if len(cls) < used_k:
                raise ValueError(f"Class {i} does not have enough samples to pick {k} items.")

            # Calculate weights inversely proportional to the selection count
            weights = 1.0 / (1.0 + selection_counts[i])
            weights /= weights.sum()

            # Choose k elements based on weights
            selected_indices = rng.choice(len(cls), size=used_k, replace=False, p=weights)
            selection_counts[i][selected_indices] += 1  # Update counts

            # Add selected elements to the combined list
            combined_list.extend([cls[idx] for idx in selected_indices])

        rest_list = [full_list[i] for i in range(len(full_list)) if i not in combined_list]

        generated_lists.append(combined_list)
        rest_lists.append(rest_list)

    if return_rest_lists:
        # return the list in tuple of two for each iteration
        return_list = []
        for i in range(iterations):
            return_list.append((np.array(rest_lists[i]), np.array(generated_lists[i])))
        return return_list
    else:
        return generated_lists

def find_and_convert_large_white_region(image, img_type='tensor'):
    if img_type == 'tensor':
        max_val = torch.max(image)
        min_val = torch.min(image)

        min_max_gap = max_val - min_val
        thresh = max_val - (255 - 240) / 255 * min_max_gap
        if len(image.shape) == 3:
            gray_image = image[0].clone()
        else:
            gray_image = image.clone()
        white_mask = gray_image > thresh
        top_row = torch.argmax(torch.any(white_mask, dim=1).int())
        bottom_row = len(white_mask) - torch.argmax(torch.any(white_mask.flip(0), dim=1).int())

        connected_region = torch.zeros_like(gray_image, dtype=torch.bool)
        for row in range(top_row, bottom_row):
            first_white_pixel = np.argmax(white_mask[row].numpy())

            center_area = row > len(white_mask) * 0.2 and row < len(white_mask) - len(white_mask) * 0.2
            if first_white_pixel > 0.01 * len(white_mask[row]):
                pass
            else:
                for col in range(first_white_pixel, len(white_mask[row])):
                    if not white_mask[row, col]:
                        if col > 0.05 * len(white_mask[row]):
                            connected_region[row, col:col+5] = True
                        break
                    connected_region[row, col] = True
            last_white_pixel = len(white_mask[row]) - torch.argmax(white_mask[row].flip(0).int()) - 1
            if last_white_pixel < len(white_mask[row]) - 0.01 * len(white_mask[row]):
                pass
            else:
                for col in range(last_white_pixel, -1, -1):
                    if not white_mask[row, col]:
                        if col < len(white_mask[row]) - 0.05 * len(white_mask[row]):
                            connected_region[row, col:col-5] = True
                        break
                    connected_region[row, col] = True
        gray_image[connected_region] = 0
        if len(image.shape) == 3:
            return_img = torch.repeat_interleave(gray_image.unsqueeze(0), image.shape[0], dim=0)
        else:
            return_img = gray_image

        return return_img

    elif img_type == 'pil':
        # Convert image to grayscale
        gray_image = np.array(image.convert('L'))

        # Threshold the image to find white regions
        white_mask = gray_image > 240


        # Find top and bottom rows that are all white pixels
        top_row = np.argmax(np.any(white_mask, axis=1))

        bottom_row = len(white_mask) - np.argmax(np.any(white_mask[::-1], axis=1))

        # Initialize connected region
        connected_region = np.zeros_like(gray_image, dtype=bool)

        # Scan from top_row to bottom_row
        for row in range(top_row, bottom_row):
            # Find the first white pixel in the row
            first_white_pixel = np.argmax(white_mask[row])
            center_area = row > len(white_mask) * 0.2 and row < len(white_mask) - len(white_mask) * 0.2
            if first_white_pixel > 0.01 * len(white_mask[row]):
                pass
            else:

                # Expand the connected region horizontally until the first non-white pixel
                for col in range(first_white_pixel, len(white_mask[row])):
                    if not white_mask[row, col]:
                        if col > 0.05 * len(white_mask[row]):
                            connected_region[row, col:col+5] = True
                        break
                    connected_region[row, col] = True

            # Find the last white pixel in the row
            last_white_pixel = len(white_mask[row]) - np.argmax(white_mask[row][::-1]) - 1
            if last_white_pixel < len(white_mask[row]) - 0.01 * len(white_mask[row]):
                pass
            else:
                # Expand the connected region horizontally until the first non-white pixel
                for col in range(last_white_pixel, -1, -1):
                    if not white_mask[row, col]:
                        if col < len(white_mask[row]) - 0.05 * len(white_mask[row]):
                            connected_region[row, col:col-5] = True
                        break
                    connected_region[row, col] = True

        # Convert the connected region to black
        gray_image[connected_region] = 0

        return Image.fromarray(gray_image)



def get_mask(x, img_type='3D', high_res=False, high_res_input_size = (20, 32, 32), input_size=(20, 16, 16), p_emb_mask_ratio=0.6, patch_size=16, pre_masks=None, return_tensor=True):

    x_norm = F.normalize(x.float(), p=2, dim=2)
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))

    actual_hwt = cosine_similarity.shape[1]

    if img_type == '3D':
        if high_res:
            pos_hw = high_res_input_size[1] * high_res_input_size[2]
            temporal = actual_hwt // pos_hw
            high_res_input_size = (temporal, high_res_input_size[1], high_res_input_size[2])
            h = int(high_res_input_size[1])
        else:
            pos_hw = input_size[1] * input_size[2]
            temporal = actual_hwt // pos_hw
            input_size = (temporal, input_size[1], input_size[2])
            h = int(input_size[1])

        diag_cosine_similarity_list = [torch.stack([cosine_similarity[n, i*pos_hw:(i+1)*pos_hw, i*pos_hw:(i+1)*pos_hw] for i in range(temporal)]) for n in range(cosine_similarity.shape[0])]

        num_frames = temporal
        hw = pos_hw

    fill_gap = 5 if high_res else 2
    up_down_clear = 6 if high_res else 3
    fill_bottom_center = True if not high_res else False
    batched_filled_mask_list = []
    s_size = int(pos_hw ** 0.5)
    top_k = int(pos_hw * p_emb_mask_ratio)
    for b in range(cosine_similarity.shape[0]):
        diag_cosine_similarity = diag_cosine_similarity_list[b]
        unmask_ratio = np.zeros(num_frames)
        adjusted_mask_list = np.zeros((num_frames, s_size, s_size))
        for n in range(num_frames):

            mask_cos = diag_cosine_similarity[n]
            sum_cosine_similarity = mask_cos.sum(dim=1) / pos_hw


            _, indices = torch.topk(sum_cosine_similarity, top_k)


            patched_imgs = torch.zeros(s_size, s_size)
            j_indices = indices % s_size
            i_indices = indices // s_size
            patched_imgs[i_indices, j_indices] = 1


            adjusted_mask, col_median = process_and_adjust_mask(patched_imgs, pos_idx=h, fill_gap=fill_gap, up_down_clear=up_down_clear, fill_bottom_center=fill_bottom_center)
            ratio_unmasked = 1 - np.sum(adjusted_mask) / (pos_hw)
            unmask_ratio[n] = ratio_unmasked
            adjusted_mask_list[n] = adjusted_mask


        max_unmask_ratio = np.max(unmask_ratio)
        max_to_unmask_number = int(pos_hw * max_unmask_ratio)
        max_to_mask_number = pos_hw - max_to_unmask_number
        anchor_num_mask = pos_hw // 2
        actual_to_mask_number = min(max_to_mask_number, anchor_num_mask)

        filled_mask_list = torch.zeros(num_frames, s_size, s_size)
        for n in range(num_frames):
            adjusted_mask = adjusted_mask_list[n]
            filled_mask = fill_patch_mask_to_ratio(adjusted_mask, to_mask_number=actual_to_mask_number)

            filled_mask_list[n] = torch.tensor(filled_mask)

        batched_filled_mask_list.append(filled_mask_list)
    return batched_filled_mask_list


def process_and_adjust_mask(mask, pos_idx=16, fill_gap=2, up_down_clear=3, fill_bottom_center=False, fill_holes=False):
    adjusted_mask = np.copy(mask)

    col_median = np.zeros(mask.shape[1])
    for j in range(mask.shape[1]):
        col = adjusted_mask[:, j]

        boundary_dist = int(pos_idx // 8)


        if j >= boundary_dist and j <= pos_idx - boundary_dist:

            adjusted_mask[(pos_idx-up_down_clear):, j] = 1
            longest_start, longest_end = find_longest_zero_sequence(col)
            if longest_start > up_down_clear:
                adjusted_mask[:up_down_clear, j] = 1

        idx_list_pos_is_0 = np.where(col == 0)[0]
        if len(idx_list_pos_is_0) == 0:
            continue
        median = round(np.median(idx_list_pos_is_0))

        col_median[j] = median
        adjusted_mask[median, j] = 0
        for k in range(fill_gap):

            if median - 1 - k >= 0:
                adjusted_mask[median - k - 1, j] = 0
            if median + 1 + k < mask.shape[0]:
                adjusted_mask[median + k + 1, j] = 0


    if fill_bottom_center:
        set_bottom_flag = False
        up_mask = adjusted_mask[:, boundary_dist:pos_idx - boundary_dist]

        col_bottom = np.zeros(up_mask.shape[1])
        for bottom in range(pos_idx - up_down_clear, pos_idx//2 , -1):
            row = up_mask[bottom]
            for j in range(len(row)):

                if row[j] == 0 and col_bottom[j] == 0:

                    col_bottom[j] = bottom

            if sum(row) / len(row) < 0.2:
                set_bottom_flag = True
                adjusted_mask[bottom] = 0
                for j in range(boundary_dist, pos_idx - boundary_dist):
                    adjusted_mask[bottom-fill_gap-1:bottom, j] = 0
            if set_bottom_flag:
                break

        if set_bottom_flag == False and np.sum(col_bottom) > 0:
            max_col_bottom = int(np.max(col_bottom))
            for j in range(len(col_bottom)):
                if col_bottom[j] == max_col_bottom:
                    col_bottom[j] -= 1
            max_col_bottom_new = int(np.max(col_bottom))

            sum_max = 0
            for j in range(len(col_bottom)):
                if col_bottom[j] == max_col_bottom_new:
                    sum_max += 1
            if sum_max / len(col_bottom) > 0.8:

                adjusted_mask[max_col_bottom] = 0
                for j in range(boundary_dist, pos_idx - boundary_dist):
                    adjusted_mask[max_col_bottom-fill_gap-1:max_col_bottom, j] = 0


    return adjusted_mask, col_median


def fill_patch_mask_to_ratio(mask, to_mask_number=None):
    num_masked = np.sum(mask)
    if to_mask_number is None:
        to_mask_number = mask.shape[0] * mask.shape[1] // 2
    if num_masked <= to_mask_number:
        return mask

    else:
        num_to_unmask = num_masked - to_mask_number
    top_idx = np.zeros(mask.shape[1], dtype=np.int32)
    bottom_idx = np.zeros(mask.shape[1], dtype=np.int32)
    bg_dist = mask.shape[1] // 8
    filled_mask = np.copy(mask)
    for j in range(mask.shape[1]):

        top_idx[j], bottom_idx[j] = find_longest_zero_sequence(mask[:, j])

    number_filled = 0
    fill_idx_start_idx = np.copy(bottom_idx) + 1

    index_iter_type = np.zeros(mask.shape[1]) # 0: bottom, 1: top
    while number_filled < num_to_unmask:
        has_unmask_option = False
        for j in range(bg_dist, mask.shape[0] - bg_dist):
            select_idx = fill_idx_start_idx[j]

            if select_idx >= mask.shape[0]:
                select_idx = top_idx[j] - 1
                index_iter_type[j] = 1
                if select_idx < 0:
                    index_iter_type[j] = -1

            if filled_mask[select_idx, j] == 0:
                if index_iter_type[j] == -1:
                    continue
                elif index_iter_type[j] == 1:
                    fill_idx_start_idx[j] -= 1
                    if fill_idx_start_idx[j] < 0:
                        index_iter_type[j] = -1
                elif index_iter_type[j] == 0:
                    fill_idx_start_idx[j] += 1
                    if fill_idx_start_idx[j] >= mask.shape[0]:
                        index_iter_type[j] = 1
                        fill_idx_start_idx[j] = top_idx[j] - 1
                        if fill_idx_start_idx[j] < 0:
                            index_iter_type[j] = -1

            if filled_mask[select_idx, j] == 1 and index_iter_type[j] != -1:

                filled_mask[select_idx, j] = 0
                number_filled += 1
                has_unmask_option = True
                if index_iter_type[j] == 0:
                    fill_idx_start_idx[j] += 1
                elif index_iter_type[j] == 1:
                    fill_idx_start_idx[j] -= 1
                if fill_idx_start_idx[j] >= mask.shape[0]:
                    fill_idx_start_idx[j] = top_idx[j] - 1
                    index_iter_type[j] = 1

                if fill_idx_start_idx[j] < 0:
                    index_iter_type[j] = -1

                if number_filled >= num_to_unmask:
                    break

    return filled_mask


def find_longest_zero_sequence(nums):
    longest_start = 0
    longest_length = 0
    current_start = None
    current_length = 0

    for i, num in enumerate(nums):
        if num == 0:
            if current_start is None:
                current_start = i
            current_length += 1
        else:
            if current_length > longest_length:
                longest_start = current_start
                longest_length = current_length
            current_start = None
            current_length = 0

    # Check at the end of the loop in case the longest sequence ends at the last element
    if current_length > longest_length:
        longest_start = current_start
        longest_length = current_length

    if longest_length == 0:
        return (0, 0)  # or return (-1, -1) if no zero sequence exists

    return (longest_start, longest_start + longest_length - 1)

def str_to_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List of integers expected.")

def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]



def show_image(image, title=''):
    # image is [3, H, W]

    if len(image.shape) == 3:
        image = image[0]
    assert len(image.shape) == 2
    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


# added by zucks
def load_model_retfound(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        print(model_without_ddp.state_dict()['pos_embed_spatial'].shape)
        read_in_q_k_v(checkpoint['model'], num_hidden_layers=24, hidden_size=1024)
        read_in_q_k_v(checkpoint['model'], num_hidden_layers=8, hidden_size=512, prefix='decoder_')
        interpolate_pos_embed_2Dto3D(model_without_ddp, checkpoint['model'])
        convert_patchembed_2Dto3D(checkpoint['model'])
        msg, msg2 = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        print("Missing keys: ", msg)
        print("Unexpected keys: ", msg2)



# added by zucks 0710 help load retfound for enface
def load_model_retfound_flash_attn_2d(args, model_without_ddp, convert_pos_embed=True, high_res_model=False, encoder_only=False, preload_model=None):
    if args.resume or preload_model is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif preload_model is not None:
            checkpoint = {'model': preload_model.state_dict()}
            print('preload model: ', preload_model.state_dict().keys())
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print('checkpoint keys: ', checkpoint['model'].keys())
        print('model keys: ', model_without_ddp.state_dict().keys())
        print(checkpoint['model']['blocks.0.attn.qkv.weight'].shape)
        print(checkpoint['model']['pos_embed'].shape)
        print(model_without_ddp.state_dict()['pos_embed'].shape)
        interpolate_pos_embed(model_without_ddp, checkpoint['model'])
        print('After interpolate pos embed')
        print(checkpoint['model']['pos_embed'].shape)
        print(model_without_ddp.state_dict()['pos_embed'].shape)

        if high_res_model:
            checkpoint['model']['high_res_patch_embed.proj.weight'] = checkpoint['model']['patch_embed.proj.weight']
            checkpoint['model']['high_res_patch_embed.proj.bias'] = checkpoint['model']['patch_embed.proj.bias']

        msg, msg2 = model_without_ddp.load_state_dict_to_backbone_retfound(checkpoint['model'], strict=False, encoder_only=encoder_only)
        # msg, msg1 = model_without_ddp.load_state_dict_to_backbone(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        print("Missing keys: ", msg)
        print("Unexpected keys: ", msg2)


# added by zucks 0626 help load retfound and imagenet for 3D
def load_model_retfound_flash_attn_3d(args, model_without_ddp, convert_pos_embed=True, high_res_model=True, encoder_only=False, preload_model=None):
    if args.resume or preload_model is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif preload_model is not None:
            checkpoint = {'model': preload_model.state_dict()}
            print('preload model: ', preload_model.state_dict().keys())
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if convert_pos_embed:

            interpolate_pos_embed_2Dto3D(model_without_ddp, checkpoint['model'], high_res_patch_embed=high_res_model)
        convert_patchembed_2Dto3D(checkpoint['model'])
        if high_res_model:
            checkpoint['model']['high_res_patch_embed.proj.weight'] = checkpoint['model']['patch_embed.proj.weight']
            checkpoint['model']['high_res_patch_embed.proj.bias'] = checkpoint['model']['patch_embed.proj.bias']

        msg, msg2 = model_without_ddp.load_state_dict_to_backbone_retfound(checkpoint['model'], strict=False, encoder_only=encoder_only)

        print("Resume checkpoint %s" % args.resume)
        print("Missing keys: ", msg)
        print("Unexpected keys: ", msg2)

# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, num_hidden_layers, hidden_size, prefix=""):
    for i in range(num_hidden_layers):

        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}blocks.{i}.attn.q.weight"] = in_proj_weight[
            : hidden_size, :
        ]
        state_dict[f"{prefix}blocks.{i}.attn.q.bias"] = in_proj_bias[: hidden_size]
        state_dict[f"{prefix}blocks.{i}.attn.k.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"{prefix}blocks.{i}.attn.k.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]
        state_dict[f"{prefix}blocks.{i}.attn.v.weight"] = in_proj_weight[
            -hidden_size :, :
        ]
        state_dict[f"{prefix}blocks.{i}.attn.v.bias"] = in_proj_bias[-hidden_size :]

def convert_patchembed_2Dto3D(state_dict):
    in_proj_weight = state_dict.pop("patch_embed.proj.weight")
    new_in_project_weight = in_proj_weight.unsqueeze(1)
    state_dict["patch_embed.proj.weight"] = new_in_project_weight


def interpolate_pos_embed_2Dto3D(model, checkpoint_model, high_res_patch_embed=False):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        print("pos_embed_checkpoint shape: ", pos_embed_checkpoint.shape)
        embedding_size = pos_embed_checkpoint.shape[-1]
        cls_token, pos_embed_spatial = torch.split(pos_embed_checkpoint, [1, 196], dim=1) # This only works for retfound and imagenet_224_vitlarge
        print("cls_token shape: ", cls_token.shape, "pos_embed_spatial shape: ", pos_embed_spatial.shape)
        num_patches = model.patch_embed.num_patches // (model.pred_t_dim // model.t_pred_patch_size)
        if high_res_patch_embed:
            num_patches = num_patches * 4
        num_extra_tokens = model.pos_embed_spatial.shape[-2] - num_patches
        print(f"model.patch_embed.num_patches: {model.patch_embed.num_patches}, model.pred_t_dim: {model.pred_t_dim}, model.t_pred_patch_size: {model.t_pred_patch_size}")
        print(f"num_patches: {num_patches}, num_extra_tokens: {num_extra_tokens}")
        print(f"num_extra_tokens: {num_extra_tokens}", model.pos_embed_spatial.shape)
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_spatial.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_spatial[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            # print("new_pos_embed shape: ", new_pos_embed.shape)
            checkpoint_model["pos_embed_spatial"] = new_pos_embed
            checkpoint_model["pos_embed_class"] = cls_token
            checkpoint_model.pop("pos_embed")

    if "decoder_pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["decoder_pos_embed"]
        # print("decoder_pos_embed shape: ", pos_embed_checkpoint.shape)
        embedding_size = pos_embed_checkpoint.shape[-1]
        cls_token, pos_embed_spatial = torch.split(pos_embed_checkpoint, [1, 196], dim=1)
        # print("cls_token shape: ", cls_token.shape, "pos_embed_spatial shape: ", pos_embed_spatial.shape)
        num_patches = model.patch_embed.num_patches // (model.pred_t_dim // model.t_pred_patch_size)
        if high_res_patch_embed:
            num_patches = num_patches * 4
        num_extra_tokens = model.decoder_pos_embed_spatial.shape[-2] - num_patches

        orig_size = int((pos_embed_spatial.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_spatial[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            # print("new_pos_embed shape: ", new_pos_embed.shape)
            checkpoint_model["decoder_pos_embed_spatial"] = new_pos_embed
            checkpoint_model["decoder_pos_embed_class"] = cls_token
            checkpoint_model.pop("decoder_pos_embed")



def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            print("new_pos_embed shape: ", new_pos_embed.shape)
            checkpoint_model["pos_embed"] = new_pos_embed

    if "decoder_pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["decoder_pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.decoder_pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            print("new_pos_embed shape: ", new_pos_embed.shape)
            checkpoint_model["decoder_pos_embed"] = new_pos_embed

# added by zucks
def interpolate_temporal_pos_embed(model, checkpoint_model, smaller_interpolate_type='interp'):
    # assume model is vit for downstream tasks
    # [TODO]: assume no extra tokens, if needed, need to add
    if "pos_embed_temporal" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed_temporal"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_temporal_patches = pos_embed_checkpoint.shape[-2]
        new_num_temporal_patches = model.patch_embed.frames // model.patch_embed.t_patch_size

        if orig_num_temporal_patches != new_num_temporal_patches:
            print(
                "Position interpolate from %d to %d"
                % (orig_num_temporal_patches, new_num_temporal_patches)
            )

            pos_tokens = pos_embed_checkpoint.permute(0, 2, 1)

            if orig_num_temporal_patches > new_num_temporal_patches and smaller_interpolate_type == "crop":
                # crop in the middle
                start_idx = (orig_num_temporal_patches - new_num_temporal_patches) // 2
                pos_tokens = pos_tokens[:, :, start_idx:start_idx + new_num_temporal_patches]
                print(f"Crop in the middle, from {start_idx} to {start_idx + new_num_temporal_patches}")
            else:
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=new_num_temporal_patches,
                    mode="linear",
                    align_corners=False,
                )

            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = pos_tokens

            checkpoint_model["pos_embed_temporal"] = new_pos_embed


gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))

default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    tf.ToTensor(),
    gray2rgb
])