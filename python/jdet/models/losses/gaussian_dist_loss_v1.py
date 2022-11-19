from copy import deepcopy
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


def gwd_loss(pred, target, fun='sqrt', tau=2.0, reduction='mean', avg_factor=None):
    """Gaussian Wasserstein distance loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
    Returns:
        loss (jittor.Var)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    xy_distance = (mu_p - mu_t).sqr().sum(dim=-1)

    whr_distance = diag3d(sigma_p).sum(-1)
    whr_distance += diag3d(sigma_t).sum(-1)

    _t_tr = diag3d(nn.bmm(sigma_p, sigma_t)).sum(dim=-1)
    _t_det_sqrt = (jt.linalg.det(sigma_p) * jt.linalg.det(sigma_t)).clamp(0).sqrt()
    whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

    dis = xy_distance + whr_distance
    gwd_dis = dis.clamp(min_v=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + jt.sqrt(gwd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + jt.log(1 + gwd_dis))
    else:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        loss = jt.log(1 + jt.sqrt(gwd_dis) / scale)

    return reduce_loss(loss, reduction, avg_factor)


def bcd_loss(pred, target, fun='log1p', tau=1.0, reduction='mean', avg_factor=None):
    """Bhatacharyya distance loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
    Returns:
        loss (jittor.Var)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2)
    mu_t = mu_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    delta = (mu_p - mu_t).unsqueeze(-1)
    sigma = 0.5 * (sigma_p + sigma_t)
    sigma_inv = jt.linalg.inv(sigma)

    term1 = jt.log(jt.linalg.det(sigma) /
                  (jt.sqrt(jt.linalg.det(sigma_t.matmul(sigma_p))))).reshape(-1, 1)
    term2 = delta.transpose(-1, -2).matmul(sigma_inv).matmul(delta).squeeze(-1)
    dis = 0.5 * term1 + 0.125 * term2
    bcd_dis = dis.clamp(min_v=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + jt.sqrt(bcd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + jt.log(1 + bcd_dis))
    else:
        loss = 1 - 1 / (tau + bcd_dis)
        
    return reduce_loss(loss, reduction, avg_factor)


def kld_loss(pred, target, fun='log1p', tau=1.0, reduction='mean', avg_factor=None):
    """Kullback-Leibler Divergence loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
    Returns:
        loss (jittor.Var)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2)
    mu_t = mu_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    delta = (mu_p - mu_t).unsqueeze(-1)
    sigma_t_inv = jt.linalg.inv(sigma_t)
    term1 = delta.transpose(-1,
                            -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
    term2 = diag3d(sigma_t_inv.matmul(sigma_p)).sum(dim=-1, keepdims=True) + \
            jt.log(jt.linalg.det(sigma_t) / jt.linalg.det(sigma_p)).reshape(-1, 1)
    dis = term1 + term2 - 2
    kl_dis = dis.clamp(min_v=1e-6)

    if fun == 'sqrt':
        kl_loss = 1 - 1 / (tau + jt.sqrt(kl_dis))
    else:
        kl_loss = 1 - 1 / (tau + jt.log(1 + kl_dis))

    return reduce_loss(kl_loss, reduction, avg_factor)


@LOSSES.register_module()
class GDLoss_v1(nn.Module):
    """Gaussian based loss.
    Args:
        loss_type (str):  Type of loss.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Returns:
        loss (jittor.Var)
    """
    BAG_GD_LOSS = {'kld': kld_loss, 'bcd': bcd_loss, 'gwd': gwd_loss}

    def __init__(self,
                 loss_type,
                 fun='sqrt',
                 tau=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GDLoss_v1, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'sqrt', '']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = xy_wh_r_2_xy_sigma
        self.fun = fun
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Execute function.
        Args:
            pred (jittor.Var): Predicted convexes.
            target (jittor.Var): Corresponding gt convexes.
            weight (jittor.Var): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not jt.any(weight > 0)) and (reduction != 'none'):
            mask = (weight > 0).detach()
            return (pred[mask] * weight[mask].reshape(-1, 1)).sum()
        if weight is not None and weight.ndim > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        mask = (weight > 0)
        pred = pred[mask]
        target = target[mask]
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred, target, fun=self.fun, tau=self.tau,
            reduction=reduction, avg_factor=avg_factor, **
            _kwargs) * self.loss_weight
