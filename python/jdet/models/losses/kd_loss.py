# Copyright (c) OpenMMLab. All rights reserved.
import jittor as jt 
from jittor import nn 
from jdet.utils.registry import LOSSES
import warnings

def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       weight,
                                       Tem=1,
                                       reduction="mean",
                                       avg_factor=None,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    if weight is not None:
        pos = weight.reshape(-1) > 0
        if pos.sum()>0:
            pred = pred[pos,:]
            soft_label = soft_label[pos,:]
        else: 
            kd_loss = pred.sum()*0
    target = jt.nn.softmax(soft_label / Tem, dim=1)
    if detach_target:
        target = target.detach()
    kl_div = jt.nn.KLDivLoss(log_target=False)
    kd_loss = kl_div(jt.nn.softmax(pred / Tem, dim=1).log(), target) * (Tem * Tem)

    return kd_loss

@LOSSES.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def execute(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            detach_target=True,
            Tem=self.T)

        return loss_kd

def im_loss(x, soft_target, reduction="mean"):
    return jt.nn.mse_loss(x, soft_target)

@LOSSES.register_module()
class IMLoss(nn.Module):
    """Loss function for feature imitation using MSE loss.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                x,
                soft_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_im = self.loss_weight * im_loss(
            x, soft_target, reduction=reduction)

        return loss_im
