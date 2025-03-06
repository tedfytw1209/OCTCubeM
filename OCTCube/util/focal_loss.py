
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='none', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
        # print(input, target)
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # print(logpt.shape, logpt)
        pt = torch.exp(logpt[:, 1:])

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt[:, 1:]
        print(focal_loss.detach().cpu().numpy(), focal_loss.mean().item(), logpt[:, 0].item())
        # print(len(logpt), logpt.shape[1], )
        balanced_focal_loss = (self.balance_param * focal_loss.mean() * (logpt.shape[1] - 1)  - logpt[:, 0]) / logpt.shape[1]
        return balanced_focal_loss