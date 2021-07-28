import jittor as jt 
from jittor import nn
from jdet.utils.registry import LOSSES

def binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, reduction="none"):
    
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(jt.safe_log(jt.maximum((-max_val).exp()+(-output - max_val).exp(),1e-10))+max_val))
    else:
        loss = (1-target)*output+max_val+jt.safe_log(jt.maximum((-max_val).exp()+(-output -max_val).exp(),1e-10))
    if weight is not None:
        loss *=weight.broadcast(loss,[1])

    if reduction=="mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
def sigmoid_cross_entropy_with_logits(logits, labels):
    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    relu_logits = jt.ternary(logits >= 0., logits, jt.broadcast_var(0.0, logits))
    neg_abs_logits = -jt.abs(logits)
    return relu_logits - logits * labels + jt.log((neg_abs_logits).exp() + 1)


def sigmoid_focal_loss(inputs,targets,weight=None, alpha = -1,gamma = 2,reduction = "none",avg_factor=None):    
    targets = targets.broadcast(inputs,[1])
    targets = (targets.index(1)+1)==targets
    p = inputs.sigmoid()
    # assert(weight is None)
    # ce_loss = sigmoid_cross_entropy_with_logits(inputs, targets)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets,weight, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        if avg_factor is None:
            avg_factor = loss.numel()
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss



def mmcv_sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
    loss = _sigmoid_focal_loss(pred.contiguous(), target, gamma, alpha, None,
                               'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def jdet_sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    from jdet.models.losses._focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def test_sigmoid_focal_loss():
    (inputs,targets,weight,alpha,gamma,reduction,avg_factor) = jt.load("test_focal_loss.pkl")
    from jdet.models.losses._focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
    loss = jdet_sigmoid_focal_loss(inputs,targets,None,alpha,gamma,reduction,avg_factor)
    print(loss)
    print(jt.grad(loss,inputs).sum())
    import torch 
    inputs = torch.from_numpy(inputs.numpy()).cuda()
    inputs = inputs.requires_grad_()
    mmcv_loss = mmcv_sigmoid_focal_loss(inputs,torch.LongTensor(targets.numpy()).cuda(),None,alpha,gamma,reduction,avg_factor)
    print(mmcv_loss)
    mmcv_loss.backward()
    print(inputs.grad.sum())


def test_focal_loss():
    import pickle 
    fam_cls_score,labels,label_weights,num_total_samples = pickle.load(open(f"/home/lxl/workspace/s2anet/fam_cls_tmp_8.pkl","rb"))
    fam_cls_score = jt.array(fam_cls_score)
    labels = jt.array(labels).int()
    label_weights = jt.array(label_weights)
    focal_loss = FocalLoss(gamma=1)
    output = focal_loss(fam_cls_score,labels,None,1)
    print(output)
    print(jt.grad(output,fam_cls_score))


if __name__ == "__main__":
    jt.flags.use_cuda=1
    # test_sigmoid_focal_loss()
    test_focal_loss()