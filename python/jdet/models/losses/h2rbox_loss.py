from jdet.utils.registry import LOSSES, build_from_cfg
import jittor as jt
from jittor import nn


@LOSSES.register_module()
class H2RBoxLoss(nn.Module):
    def __init__(self, center_loss_cfg, shape_loss_cfg, angle_loss_cfg,
                 reduction='mean', loss_weight=1.0):
        super(H2RBoxLoss, self).__init__()
        self.center_loss = build_from_cfg(center_loss_cfg, LOSSES)
        self.shape_loss = build_from_cfg(shape_loss_cfg, LOSSES)
        self.angle_loss = build_from_cfg(angle_loss_cfg, LOSSES)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted boxes.
            target (torch.Tensor): Corresponding gt boxes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        xy_pred = pred[..., :2]
        xy_target = target[..., :2]
        hbb_pred1 = jt.concat([-pred[..., 2:4], pred[..., 2:4]], dim=-1)
        hbb_pred2 = hbb_pred1[..., [1, 0, 3, 2]]
        hbb_target = jt.concat([-target[..., 2:4], target[..., 2:4]], dim=-1)
        d_a_pred = pred[..., 4] - target[..., 4]

        center_loss = self.center_loss(xy_pred, xy_target,
                                       weight=weight[:, None],
                                       reduction_override=reduction,
                                       avg_factor=avg_factor)
        shape_loss1 = self.shape_loss(hbb_pred1, hbb_target,
                                      weight=weight,
                                      reduction_override=reduction,
                                      avg_factor=avg_factor) + self.angle_loss(
            d_a_pred.sin(), jt.zeros_like(d_a_pred), weight=weight,
            reduction_override=reduction, avg_factor=avg_factor)
        shape_loss2 = self.shape_loss(hbb_pred2, hbb_target,
                                      weight=weight,
                                      reduction_override=reduction,
                                      avg_factor=avg_factor) + self.angle_loss(
            d_a_pred.cos(), jt.zeros_like(d_a_pred), weight=weight,
            reduction_override=reduction, avg_factor=avg_factor)
        loss_bbox = center_loss + jt.minimum(shape_loss1, shape_loss2)
        return self.loss_weight * loss_bbox
