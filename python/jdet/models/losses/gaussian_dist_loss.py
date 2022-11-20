# Copyright (c) SJTU. All rights reserved.
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


def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss.
    Args:
        distance (jittor.Var)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
    Returns:
        loss (jittor.Var)
    """
    if fun == 'log1p':
        distance = jt.log(1 + distance)
    elif fun == 'sqrt':
        distance = jt.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance



def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True, reduction='mean', avg_factor=None):
    """Gaussian Wasserstein distance loss.
    Derivation and simplification:
        Given any positive-definite symmetrical 2*2 matrix Z:
            :math:`Tr(Z^{1/2}) = λ_1^{1/2} + λ_2^{1/2}`
        where :math:`λ_1` and :math:`λ_2` are the eigen values of Z
        Meanwhile we have:
            :math:`Tr(Z) = λ_1 + λ_2`
            :math:`det(Z) = λ_1 * λ_2`
        Combination with following formula:
            :math:`(λ_1^{1/2}+λ_2^{1/2})^2 = λ_1+λ_2+2 *(λ_1 * λ_2)^{1/2}`
        Yield:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
        For gwd loss the frustrating coupling part is:
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σp^{1/2})^{1/2})`
        Assuming :math:`Z = Σ_p^{1/2} * Σ_t * Σ_p^{1/2}` then:
            :math:`Tr(Z) = Tr(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = Tr(Σ_p^{1/2} * Σ_p^{1/2} * Σ_t)
            = Tr(Σ_p * Σ_t)`
            :math:`det(Z) = det(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = det(Σ_p^{1/2}) * det(Σ_t) * det(Σ_p^{1/2})
            = det(Σ_p * Σ_t)`
        and thus we can rewrite the coupling part as:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σ_p^{1/2})^{1/2})
            = (Tr(Σ_p * Σ_t) + 2 * (det(Σ_p * Σ_t))^{1/2})^{1/2}`
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.
    Returns:
        loss (jittor.Var)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = (xy_p - xy_t).sqr().sum(dim=-1)

    whr_distance = diag3d(Sigma_p).sum(-1)
    whr_distance += diag3d(Sigma_t).sum(-1)

    _t_tr = diag3d(nn.bmm(Sigma_p, Sigma_t)).sum(dim=-1)
    _t_det_sqrt = (jt.linalg.det(Sigma_p) * jt.linalg.det(Sigma_t)).clamp(0).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    loss = postprocess(distance, fun=fun, tau=tau)
    return reduce_loss(loss, reduction, avg_factor)


def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, reduction='mean', avg_factor=None):
    """Kullback-Leibler Divergence loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.
    Returns:
        loss (jittor.Var)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = jt.linalg.inv(Sigma_p)
    Sigma_p_inv = Sigma_p_inv / jt.linalg.det(Sigma_p).unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance =  0.5 * nn.bmm(nn.bmm(dxy.permute(0, 2, 1), Sigma_p_inv), dxy).view(-1)

    whr_distance = 0.5 * diag3d(nn.bmm(Sigma_p_inv, Sigma_t)).sum(dim=-1)
    Sigma_p_det_log = jt.log(jt.linalg.det(Sigma_p))
    Sigma_t_det_log = jt.log(jt.linalg.det(Sigma_t))
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])

    loss = postprocess(distance, fun=fun, tau=tau)
    return reduce_loss(loss, reduction, avg_factor)


def jd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, reduction='mean', avg_factor=None):
    """Symmetrical Kullback-Leibler Divergence loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.
    Returns:
        loss (jittor.Var)
    """
    jd = kld_loss(
        pred,
        target,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=False,
        reduction='none')
    jd = jd + kld_loss(
        target,
        pred,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=False,
        reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(1e-7).sqrt()
    loss = postprocess(jd, fun=fun, tau=tau)
    return reduce_loss(loss, reduction, avg_factor)


def kld_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, reduction='mean', avg_factor=None):
    """Symmetrical Max Kullback-Leibler Divergence loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.
    Returns:
        loss (jittor.Var)
    """
    kld_pt = kld_loss(
        pred,
        target,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_tp = kld_loss(
        target,
        pred,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_symmax = jt.max(kld_pt, kld_tp)
    loss = postprocess(kld_symmax, fun=fun, tau=tau)
    return reduce_loss(loss, reduction, avg_factor)


def kld_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, reduction='mean', avg_factor=None):
    """Symmetrical Min Kullback-Leibler Divergence loss.
    Args:
        pred (jittor.Var): Predicted bboxes.
        target (jittor.Var): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.
    Returns:
        loss (jittor.Var)
    """
    kld_pt = kld_loss(
        pred,
        target,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_tp = kld_loss(
        target,
        pred,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_symmin = jt.min(kld_pt, kld_tp)
    loss = postprocess(kld_symmin, fun=fun, tau=tau)
    return reduce_loss(loss, reduction, avg_factor)


@LOSSES.register_module()
class GDLoss(nn.Module):
    """Gaussian based loss.
    Args:
        loss_type (str):  Type of loss.
        representation (str, optional): Coordinate System.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        alpha (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Returns:
        loss (jittor.Var)
    """
    BAG_GD_LOSS = {
        'gwd': gwd_loss,
        'kld': kld_loss,
        'jd': jd_loss,
        'kld_symmax': kld_symmax_loss,
        'kld_symmin': kld_symmin_loss
    }
    BAG_PREP = {
        'xy_wh_r': xy_wh_r_2_xy_sigma
    }

    def __init__(self,
                 loss_type,
                 representation='xy_wh_r',
                 fun='log1p',
                 tau=0.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none', 'sqrt']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
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
            weight (jittor.Var, optional): The weight of loss for each
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
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight
