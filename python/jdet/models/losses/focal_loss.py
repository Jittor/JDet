import jittor as jt 
from jittor import nn
from jdet.utils.registry import LOSSES

def binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, reduction="none"):
    
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(jt.log(jt.maximum((-max_val).exp()+(-output - max_val).exp(),1e-10))+max_val))
    else:
        loss = (1-target)*output+max_val+jt.log(jt.maximum((-max_val).exp()+(-output -max_val).exp(),1e-10))
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


