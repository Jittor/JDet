import jittor as jt 

def smooth_l1_loss(pred,target,beta=1.,weight=None,avg_factor=None,reduction="mean"):
    diff = jt.abs(pred-target)
    if beta!=0.:
        flag = (diff<beta).float()
        loss = flag*0.5* diff.sqr()/beta + (1-flag)*(diff - 0.5 * beta)
    else:
        loss = diff 

    if weight is not None:
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        return loss.sum()/avg_factor
    elif reduction == "sum":
        return loss.sum()

    return loss 

