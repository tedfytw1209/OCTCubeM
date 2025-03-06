# Revised by Zixuan Zucks Liu @University of Washington

from selectors import SelectorKey
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        enface_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_enface_features = hvd.allgather(enface_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_enface_features = hvd.allgather(enface_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_enface_features = list(all_enface_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_enface_features[rank] = enface_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_enface_features = torch.cat(gathered_enface_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_enface_features = torch.cat(torch.distributed.nn.all_gather(enface_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_enface_features = [torch.zeros_like(enface_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_enface_features, enface_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_enface_features[rank] = enface_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_enface_features = torch.cat(gathered_enface_features, dim=0)

    return all_image_features, all_enface_features


def gather_features_3mod(
        image_features,
        enface1_features,
        enface2_features,
        t_weight1,
        t_weight2,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_enface1_features = hvd.allgather(enface1_features)
            all_enface2_features = hvd.allgather(enface2_features)
            all_t_weight1 = hvd.allgather(t_weight1)
            all_t_weight2 = hvd.allgather(t_weight2)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_enface1_features = hvd.allgather(enface1_features)
                all_enface2_features = hvd.allgather(enface2_features)
                all_t_weight1 = hvd.allgather(t_weight1)
                all_t_weight2 = hvd.allgather(t_weight2)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_enface1_features = list(all_enface1_features.chunk(world_size, dim=0))
                gathered_enface2_features = list(all_enface2_features.chunk(world_size, dim=0))
                gathered_t_weight1 = list(all_t_weight1.chunk(world_size, dim=0))
                gathered_t_weight2 = list(all_t_weight2.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_enface1_features[rank] = enface1_features
                gathered_enface2_features[rank] = enface2_features
                gathered_t_weight1[rank] = t_weight1
                gathered_t_weight2[rank] = t_weight2
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_enface1_features = torch.cat(gathered_enface1_features, dim=0)
                all_enface2_features = torch.cat(gathered_enface2_features, dim=0)
                all_t_weight1 = torch.cat(gathered_t_weight1, dim=0)
                all_t_weight2 = torch.cat(gathered_t_weight2, dim=0)
    else:
        # We gather tensors from all gpus

        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_enface1_features = torch.cat(torch.distributed.nn.all_gather(enface1_features), dim=0)
            all_enface2_features = torch.cat(torch.distributed.nn.all_gather(enface2_features), dim=0)
            all_t_weight1 = torch.cat(torch.distributed.nn.all_gather(t_weight1), dim=0)
            all_t_weight2 = torch.cat(torch.distributed.nn.all_gather(t_weight2), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_enface1_features = [torch.zeros_like(enface1_features) for _ in range(world_size)]
            gathered_enface2_features = [torch.zeros_like(enface2_features) for _ in range(world_size)]
            gathered_t_weight1 = [torch.zeros_like(t_weight1) for _ in range(world_size)]
            gathered_t_weight2 = [torch.zeros_like(t_weight2) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_enface1_features, enface1_features)
            dist.all_gather(gathered_enface2_features, enface2_features)
            dist.all_gather(gathered_t_weight1, t_weight1)
            dist.all_gather(gathered_t_weight2, t_weight2)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_enface1_features[rank] = enface1_features
                gathered_enface2_features[rank] = enface2_features
                gathered_t_weight1[rank] = t_weight1
                gathered_t_weight2[rank] = t_weight2
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_enface1_features = torch.cat(gathered_enface1_features, dim=0)
            all_enface2_features = torch.cat(gathered_enface2_features, dim=0)
            all_t_weight1 = torch.cat(gathered_t_weight1, dim=0)
            all_t_weight2 = torch.cat(gathered_t_weight2, dim=0)

    return all_image_features, all_enface1_features, all_enface2_features, all_t_weight1, all_t_weight2

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            correct_label=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.correct_label = correct_label

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_corrected_label(self, enface_features_i, enface_features_j, t=10**(-100)):
        enface_exp_i = enface_features_i.detach().cpu().unsqueeze(1)
        enface_exp_j = enface_features_j.detach().cpu().unsqueeze(0)
        dist_matrix = torch.sqrt(((enface_exp_i - enface_exp_j) ** 2).sum(dim=-1))
        L = (dist_matrix <= t).int().to(enface_features_i.dtype)
        L = L.float() / torch.sum(L, dim=1, keepdim=True)
        return L.to(enface_features_j.device)

    def forward(self, image_features, enface_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_enface_features = gather_features(
                image_features, enface_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_enface_features.T
                logits_per_enface = logit_scale * enface_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_enface_features.T
                logits_per_enface = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ enface_features.T
            logits_per_enface = logit_scale * enface_features @ image_features.T

        if self.correct_label:
            # correct the label if there exists two slides come from the same reports
            if self.world_size > 1:
                if self.local_loss:
                    labels = self.get_corrected_label(enface_features, all_enface_features)
                else:
                    labels = self.get_corrected_label(all_enface_features, all_enface_features)
            else:
                labels = self.get_corrected_label(enface_features, enface_features)
        else:
            # calculated ground-truth and cache if enabled
            num_logits = logits_per_image.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
        if self.correct_label:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_enface, labels)
            ) / 2
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_enface, labels)
            ) / 2
        return total_loss


class ThreeModalityClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            correct_label=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.correct_label = correct_label

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_corrected_label(self, features_i, features_j, t=1e-10):
        features_i_exp = features_i.detach().cpu().unsqueeze(1)
        features_j_exp = features_j.detach().cpu().unsqueeze(0)

        dist_matrix = torch.sqrt(((features_i_exp - features_j_exp) ** 2).sum(dim=-1))

        L = (dist_matrix <= t).int().to(features_i.dtype)

        L = L.float() / torch.sum(L, dim=1, keepdim=True)
        return L.to(features_j.device)

    def forward(self, image_features, enface1_features, enface2_features, logit_scale, logit_scale1, logit_scale2, t_weight1, t_weight2):
        device = image_features.device

        # Handle distributed training
        if self.world_size > 1:
            # Implement gather_features as per your distributed setup
            all_image_features, all_enface1_features, all_enface2_features, all_t_weight1, all_t_weight2 = gather_features_3mod(
                image_features, enface1_features, enface2_features, t_weight1, t_weight2,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        else:
            all_image_features = image_features
            all_enface1_features = enface1_features
            all_enface2_features = enface2_features
            all_t_weight1 = t_weight1
            all_t_weight2 = t_weight2

        if self.local_loss:
            # Compute logits between all pairs
            # Image-enface1
            logits_per_image_enface1 = logit_scale * image_features @ all_enface1_features.T
            logits_per_enface1_image = logit_scale * enface1_features @ all_image_features.T

            # Image-enface2
            logits_per_image_enface2 = logit_scale1 * image_features @ all_enface2_features.T
            logits_per_enface2_image = logit_scale1 * enface2_features @ all_image_features.T

            # enface1-enface2
            logits_per_enface1_enface2 = logit_scale2 * enface1_features @ all_enface2_features.T
            logits_per_enface2_enface1 = logit_scale2 * enface2_features @ all_enface1_features.T

            # t_weight1 and t_weight2
            used_t_weight1 = t_weight1
            used_t_weight2 = t_weight2
        else:
            # Compute logits between all pairs
            # Image-enface1
            logits_per_image_enface1 = logit_scale * all_image_features @ all_enface1_features.T
            logits_per_enface1_image = logits_per_image_enface1.T

            # Image-enface2
            logits_per_image_enface2 = logit_scale1 * all_image_features @ all_enface2_features.T
            logits_per_enface2_image = logits_per_image_enface2.T

            # enface1-enface2
            logits_per_enface1_enface2 = logit_scale2 * all_enface1_features @ all_enface2_features.T
            logits_per_enface2_enface1 = logits_per_enface1_enface2.T

            # t_weight1 and t_weight2
            used_t_weight1 = all_t_weight1
            used_t_weight2 = all_t_weight2

        # Compute labels
        if self.correct_label:
            labels = self.get_corrected_label(enface1_features, all_enface1_features)

        else:
            num_logits = all_image_features.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

        # FIXME: Didn't fill in large negative for missing modalities will have small impact on the loss, as we should have masked them out, but it looks ok for now

        # Compute the losses with masking for missing modalities
        # Create masks for valid samples
        t_mask1 = used_t_weight1 > 0
        t_mask2 = used_t_weight2 > 0

        # Image-enface1 Loss
        loss_it1 = F.cross_entropy(logits_per_image_enface1, labels, reduction='none')
        loss_t1i = F.cross_entropy(logits_per_enface1_image, labels, reduction='none')
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

        # Image-enface2 Loss
        loss_it2 = F.cross_entropy(logits_per_image_enface2, labels, reduction='none')
        loss_t2i = F.cross_entropy(logits_per_enface2_image, labels, reduction='none')
        # Apply masks
        loss_it2 = loss_it2 * used_t_weight2
        loss_t2i = loss_t2i * used_t_weight2
        if used_t_weight2.sum() == 0:
            loss_it2 = torch.tensor(0.0, device=device)
            loss_t2i = torch.tensor(0.0, device=device)
        else:
            # Average over valid samples
            loss_it2 = loss_it2.sum() / used_t_weight2.sum()
            loss_t2i = loss_t2i.sum() / used_t_weight2.sum()

        # enface1-enface2 Loss
        loss_t1t2 = F.cross_entropy(logits_per_enface1_enface2, labels, reduction='none')
        loss_t2t1 = F.cross_entropy(logits_per_enface2_enface1, labels, reduction='none')
        # Apply masks (both modalities need to be present)
        used_t_weight_pair = used_t_weight1 * used_t_weight2
        loss_t1t2 = loss_t1t2 * used_t_weight_pair
        loss_t2t1 = loss_t2t1 * used_t_weight_pair
        if used_t_weight_pair.sum() == 0:
            loss_t1t2 = torch.tensor(0.0, device=device)
            loss_t2t1 = torch.tensor(0.0, device=device)
        else:
            # Average over valid samples
            loss_t1t2 = loss_t1t2.sum() / used_t_weight_pair.sum()
            loss_t2t1 = loss_t2t1.sum() / used_t_weight_pair.sum()

        # Total loss is the average of all valid losses
        total_loss = (loss_it1 + loss_t1i + loss_it2 + loss_t2i + loss_t1t2 + loss_t2t1) / 6

        return total_loss