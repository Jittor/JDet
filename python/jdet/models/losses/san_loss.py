import jittor as jt
import numpy as np
from jittor import nn
from jdet.utils.registry import LOSSES

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = jt.randperm(x.shape[0])
    x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_loss(output, target_a, target_b, lam=1.0, eps=0.0):
    w = jt.zeros_like(output).scatter_(1, target_a.unsqueeze(1), jt.array(1))
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = nn.log_softmax(output, dim=1)
    loss_a = (-w * log_prob).sum(dim=1).mean()

    w = jt.zeros_like(output).scatter_(1, target_b.unsqueeze(1), jt.array(1))
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = nn.log_softmax(output, dim=1)
    loss_b = (-w * log_prob).sum(dim=1).mean()
    return lam * loss_a + (1 - lam) * loss_b


def smooth_loss(output, target, eps=0.1):
    w = jt.zeros_like(output).scatter_(1, target.unsqueeze(1), jt.array(1))
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = nn.log_softmax(output, dim=1)
    loss = (-w * log_prob).sum(dim=1).mean()
    return loss

@LOSSES.register_module()
class SANMixUpLoss(nn.Module):
    def __init__(self, alpha=0.2, eps=0.0):
        super(SANMixUpLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.target_a, self.target_b, self.lam = None, None, None
    
    def prepare(self, input, target):
        input, self.target_a, self.target_b, self.lam = mixup_data(input, target, self.alpha)
        return input
    
    def execute(self, output, target=None):
        return mixup_loss(output, self.target_a, self.target_b, self.lam, self.eps)

@LOSSES.register_module()
class SAMSmoothLoss(nn.Module):
    def __init__(self, eps=0.1):
        super(SAMSmoothLoss).__init__()
        self.eps = eps
    
    def execute(self, output, target):
        return smooth_loss(output, target, self.eps)
