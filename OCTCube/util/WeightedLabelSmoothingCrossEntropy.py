import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing and weight.
    """
    def __init__(self, smoothing=0.1):
        super(WeightedLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing


    def forward(self, x: torch.Tensor, target: torch.Tensor, verbose: bool=False) -> torch.Tensor:
        # Create mask to exclude rows with [0,0]
        valid_mask = (target.sum(dim=-1) != 0).float()

        if valid_mask.sum() == 0:
            return x.mean() * 0

        logprobs = F.log_softmax(x, dim=-1)

        # Convert target to indices and mask out invalid rows
        target_indices = target.argmax(dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target_indices.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # Apply the valid_mask to exclude invalid rows
        loss = loss * valid_mask # * manual_mask
        if verbose:
            print('nll_loss', nll_loss, nll_loss.shape)
            print('smooth_loss', smooth_loss, smooth_loss.shape)
            print('loss', loss, loss.shape)
            print('logprobs', logprobs, logprobs.shape)
            print('target', target, target.shape)
            print('valid_mask', valid_mask, valid_mask.shape)
            print('target_indices', target_indices, target_indices.shape)
            print('loss before sum', loss)

        return loss.sum() / valid_mask.sum()

if __name__ == "__main__":
    # Test
    loss = WeightedLabelSmoothingCrossEntropy()
    x = torch.randn(3, 10)
    target = torch.zeros(3,10)
    target[0, 0] = 1
    target[1, 5] = 1
    print(x.shape, target.shape)
    print(loss(x, target))