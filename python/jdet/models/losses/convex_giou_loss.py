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
            pred (jt.Tensor): Predicted convexes.
            target (jt.Tensor): Corresponding gt convexes.
            weight (jt.Tensor, optional): The weight of loss for each
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
        jt.Tensor: Loss tensor.
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
            pred (jt.Tensor): Predicted convexes.
            target (jt.Tensor): Corresponding gt convexes.
            weight (jt.Tensor, optional): The weight of loss for each
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


@LOSSES.register_module()
class BCConvexGIoULossFuction(jt.Function):
    """The function of BCConvex GIoU loss."""

 
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
            pred (jt.Tensor): Predicted convexes.
            target (jt.Tensor): Corresponding gt convexes.
            weight (jt.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): The reduction method of the
            loss. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        """
        convex_gious, grad = reppoints_convex_giou(pred, target)

        pts_pred_all_dx = pred[:, 0::2]
        pts_pred_all_dy = pred[:, 1::2]
        pred_left_x_inds = pts_pred_all_dx.argmin(dim=1)[0].unsqueeze(0)
        pred_right_x_inds = pts_pred_all_dx.argmax(dim=1)[0].unsqueeze(0)
        pred_up_y_inds = pts_pred_all_dy.argmin(dim=1)[0].unsqueeze(0)
        pred_bottom_y_inds = pts_pred_all_dy.argmax(dim=1)[0].unsqueeze(0)

        pred_right_x = pts_pred_all_dx.gather(dim=1, index=pred_right_x_inds)
        pred_right_y = pts_pred_all_dy.gather(dim=1, index=pred_right_x_inds)

        pred_left_x = pts_pred_all_dx.gather(dim=1, index=pred_left_x_inds)
        pred_left_y = pts_pred_all_dy.gather(dim=1, index=pred_left_x_inds)

        pred_up_x = pts_pred_all_dx.gather(dim=1, index=pred_up_y_inds)
        pred_up_y = pts_pred_all_dy.gather(dim=1, index=pred_up_y_inds)

        pred_bottom_x = pts_pred_all_dx.gather(dim=1, index=pred_bottom_y_inds)
        pred_bottom_y = pts_pred_all_dy.gather(dim=1, index=pred_bottom_y_inds)
        pred_corners = jt.concat([
            pred_left_x, pred_left_y, pred_up_x, pred_up_y, pred_right_x,
            pred_right_y, pred_bottom_x, pred_bottom_y
        ],
                                 dim=-1)

        pts_target_all_dx = target[:, 0::2]
        pts_target_all_dy = target[:, 1::2]

        target_left_x_inds = pts_target_all_dx.argmin(dim=1)[0].unsqueeze(0)
        target_right_x_inds = pts_target_all_dx.argmax(dim=1)[0].unsqueeze(0)
        target_up_y_inds = pts_target_all_dy.argmin(dim=1)[0].unsqueeze(0)
        target_bottom_y_inds = pts_target_all_dy.argmax(dim=1)[0].unsqueeze(0)

        target_right_x = pts_target_all_dx.gather(
            dim=1, index=target_right_x_inds)
        target_right_y = pts_target_all_dy.gather(
            dim=1, index=target_right_x_inds)

        target_left_x = pts_target_all_dx.gather(
            dim=1, index=target_left_x_inds)
        target_left_y = pts_target_all_dy.gather(
            dim=1, index=target_left_x_inds)

        target_up_x = pts_target_all_dx.gather(dim=1, index=target_up_y_inds)
        target_up_y = pts_target_all_dy.gather(dim=1, index=target_up_y_inds)

        target_bottom_x = pts_target_all_dx.gather(
            dim=1, index=target_bottom_y_inds)
        target_bottom_y = pts_target_all_dy.gather(
            dim=1, index=target_bottom_y_inds)

        target_corners = jt.concat([
            target_left_x, target_left_y, target_up_x, target_up_y,
            target_right_x, target_right_y, target_bottom_x, target_bottom_y
        ],
                                   dim=-1)

        pts_pred_dx_mean = pts_pred_all_dx.mean(
            dim=1).reshape(-1, 1)
        pts_pred_dy_mean = pts_pred_all_dy.mean(
            dim=1).reshape(-1, 1)
        pts_pred_mean = jt.concat([pts_pred_dx_mean, pts_pred_dy_mean], dim=-1)

        pts_target_dx_mean = pts_target_all_dx.mean(
            dim=1).reshape(-1, 1)
        pts_target_dy_mean = pts_target_all_dy.mean(
            dim=1).reshape(-1, 1)
        pts_target_mean = jt.concat([pts_target_dx_mean, pts_target_dy_mean],
                                    dim=-1)

        beta = 1.0

        diff_mean = jt.abs(pts_pred_mean - pts_target_mean)
        diff_mean_loss = jt.where(diff_mean < beta,
                                     0.5 * diff_mean * diff_mean / beta,
                                     diff_mean - 0.5 * beta)
        diff_mean_loss = diff_mean_loss.sum() / len(diff_mean_loss)

        diff_corners = jt.abs(pred_corners - target_corners)
        diff_corners_loss = jt.where(
            diff_corners < beta, 0.5 * diff_corners * diff_corners / beta,
            diff_corners - 0.5 * beta)
        diff_corners_loss = diff_corners_loss.sum() / len(diff_corners_loss)

        target_aspect = AspectRatio(target)
        smooth_loss_weight = jt.exp((-1 / 4) * target_aspect)
        loss = \
            smooth_loss_weight * (diff_mean_loss.reshape(-1, 1) +
                                  diff_corners_loss.reshape(-1, 1)) + \
            1 - (1 - 2 * smooth_loss_weight) * convex_gious

        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()

        unvaild_inds = jt.nonzero((grad > 1).sum(1))[:, 0]
        grad[unvaild_inds] = 1e-6

        reduce_grad = -grad / grad.size(0) * loss_weight
        self.convex_points_grad = reduce_grad
        return loss

    def grad(self, input=None):
        """Backward function."""
        convex_points_grad = self.convex_points_grad
        return convex_points_grad, None, None, None, None, None


bc_convex_giou_loss = BCConvexGIoULossFuction.apply


@LOSSES.register_module()
class BCConvexGIoULoss(nn.Module):
    """BCConvex GIoU loss.

    Computing the BCConvex GIoU loss between a set of predicted convexes and
    target convexes.

    Args:
        reduction (str, optional): The reduction method of the loss. Defaults
            to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Return:
        jt.Tensor: Loss tensor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(BCConvexGIoULoss, self).__init__()
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
            pred (jt.Tensor): Predicted convexes.
            target (jt.Tensor): Corresponding gt convexes.
            weight (jt.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if weight is not None and not jt.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bc_convex_giou_loss(
            pred, target, weight, reduction, avg_factor, self.loss_weight)
        return loss


def AspectRatio(gt_rbboxes):
    """Compute the aspect ratio of all gts.

    Args:
        gt_rbboxes (jt.Tensor): Groundtruth polygons, shape (k, 8).

    Returns:
        ratios (jt.Tensor): The aspect ratio of gt_rbboxes, shape (k, 1).
    """
    pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)
    edge1 = jt.sqrt(
        jt.pow(pt1[..., 0] - pt2[..., 0], 2) +
        jt.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = jt.sqrt(
        jt.pow(pt2[..., 0] - pt3[..., 0], 2) +
        jt.pow(pt2[..., 1] - pt3[..., 1], 2))

    edges = jt.stack([edge1, edge2], dim=1)

    width= jt.max(edges, 1)
    height = jt.min(edges, 1)
    ratios = (width / height)
    return ratios
