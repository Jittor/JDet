import jittor as jt
from jittor import nn
from jdet.ops.reppoints_convex_iou import reppoints_convex_giou
from jdet.utils.registry import LOSSES

@LOSSES.register_module()
class ConvexGIoULossFunction(jt.Function):
    """The function of Convex GIoU loss."""

    def execute(self,
                pred,
                target,
                weight=None,
                reduction=None,
                avg_factor=None,
                loss_weight=1.0):
        """Forward function.

        Args:
            ctx:  {save_for_backward, convex_points_grad}
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): The reduction method of the
            loss. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        """
        convex_gious, grad = reppoints_convex_giou(pred, target)

        loss = 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)

        if avg_factor is None:
            avg_factor = max(loss.shape[0],1)

        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.sum() / avg_factor

        
        unvaild_inds = jt.nonzero(jt.any(grad > 1, dim=1))[:, 0]
        grad[unvaild_inds] = 1e-6

        # _reduce_grad
        reduce_grad = -grad / avg_factor * loss_weight
        self.convex_points_grad = reduce_grad
        return loss

    def grad(self, input=None):
        """Backward function."""
        convex_points_grad = self.convex_points_grad
        return convex_points_grad, None, None, None, None, None

convex_giou_loss = ConvexGIoULossFunction.apply

@LOSSES.register_module()
class ConvexGIoULoss(nn.Module):
    """Convex GIoU loss.

    Computing the Convex GIoU loss between a set of predicted convexes and
    target convexes.

    Args:
        reduction (str, optional): The reduction method of the loss. Defaults
            to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Return:
        torch.Tensor: Loss tensor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ConvexGIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if weight is not None and not jt.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * convex_giou_loss(
            pred, target, weight, reduction, avg_factor, self.loss_weight)
        return loss

