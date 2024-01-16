import torch
from torch import nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, size_average=None, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, size_average=None, reduce=None)
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class FocalLossMulti(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLossMulti, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=1)
        prob = torch.exp(log_prob)
        one_hot = F.one_hot(target, num_classes=input.shape[1])

        focal_weight = one_hot * (1 - prob) ** self.gamma
        focal_weight += (1 - one_hot) * prob ** self.gamma
        loss = - (focal_weight * log_prob).sum(dim=1)

        return loss.mean()