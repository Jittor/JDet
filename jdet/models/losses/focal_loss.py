import jittor as jt 
from jittor import nn

def binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, reduction="none"):
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(jt.log(jt.maximum((-max_val).exp()+(-output - max_val).exp(),1e-10))+max_val))
    else:
        loss = (1-target)*output+max_val+jt.log(jt.maximum((-max_val).exp()+(-output -max_val).exp(),1e-10))
    if weight is not None:
        loss *=weight

    if reduction=="mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def sigmoid_focal_loss(inputs,targets,alpha: float = -1,gamma: float = 2,reduction: str = "none"):
    p = inputs.sigmoid()
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss