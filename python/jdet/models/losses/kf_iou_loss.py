import jittor as jt
from jittor import nn
from jdet.utils.registry import LOSSES


def diag3d(x):
    return jt.stack([jt.diag(x_) for x_ in x])

def reduce_loss(loss, reduction='mean', avg_factor=None):
    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == 'mean':
        loss = loss.sum()/avg_factor
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (jittor.Var): rbboxes with shape (N, 5).

    Returns:
        xy (jittor.Var): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (jittor.Var): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(1e-7, 1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = jt.cos(r)
    sin_r = jt.sin(r)
    R = jt.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * diag3d(wh)

    sigma = nn.bmm(nn.bmm(R, S.sqr()), R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def kfiou_loss(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               reduction='mean',
               avg_factor=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (jttor.Var): Predicted bboxes.
        target (jttor.Var): Corresponding gt bboxes.
        pred_decode (jttor.Var): Predicted decode bboxes.
        targets_decode (jttor.Var): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (jttor.Var)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)

    # Smooth-L1 norm
    diff = jt.abs(xy_p - xy_t)
    xy_loss = jt.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * jt.linalg.det(Sigma_p).sqrt()
    Vb_t = 4 * jt.linalg.det(Sigma_t).sqrt()
    K = nn.bmm(Sigma_p, jt.linalg.inv(Sigma_p + Sigma_t))
    Sigma = Sigma_p - nn.bmm(K, Sigma_p)
    Vb = 4 * jt.linalg.det(Sigma).sqrt()
    Vb = jt.where(jt.isnan(Vb), jt.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -jt.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = jt.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return reduce_loss(loss, reduction, avg_factor)


@LOSSES.register_module()
class KFLoss(nn.Module):
    """Kalman filter based loss.

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (jttor.Var)
    """

    def __init__(self,
                 fun='none',
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(KFLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                pred_decode=None,
                targets_decode=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (jttor.Var): Predicted convexes.
            target (jttor.Var): Corresponding gt convexes.
            weight (jttor.Var, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (jttor.Var): Predicted decode bboxes.
            targets_decode (jttor.Var): Corresponding gt decode bboxes.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (jttor.Var)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not jt.any(weight > 0)) and (
                reduction != 'none'):
            mask = (weight > 0).detach()
            return (pred[mask] * weight[mask].reshape(-1, 1)).sum()
        if weight is not None and weight.ndim > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        mask = (weight > 0).detach()
        pred = pred[mask]
        target = target[mask]
        pred_decode = pred_decode[mask]
        targets_decode = targets_decode[mask]

        return kfiou_loss(
            pred=pred,
            target=target,
            pred_decode=pred_decode,
            targets_decode=targets_decode,
            reduction=reduction,
            fun=self.fun,
            avg_factor=avg_factor,
            **kwargs) * self.loss_weight