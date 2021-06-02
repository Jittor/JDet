import jittor as jt 

def smooth_l1_loss(pred,target,beta=1.,weight=None,avg_factor=None,reduction="mean"):
    
    diff = jt.abs(pred-target)
    flag = (diff<beta).float()
    loss = flag*0.5* diff.sqr()/beta + (1-flag)*(diff - 0.5 * beta)

    if weight is not None:
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.numel(),1)

    if reduction == "mean":
        return loss.sum()/avg_factor
    elif reduction == "sum":
        return loss.sum()

    return loss 

def l1_loss(pred,target,weight=None,avg_factor=None,reduction="mean"):
    
    loss = jt.abs(pred-target)

    if weight is not None:
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.numel(),1)

    if reduction == "mean":
        return loss.sum()/avg_factor
    elif reduction == "sum":
        return loss.sum()

    return loss 



def faster_rcnn_loss(pred_locs,gt_locs,gt_labels,beta):
    weights = jt.zeros(pred_locs.shape)
    weights[gt_labels>0,:]=1
    n_samples = (gt_labels>=0).sum().float()
    loss = smooth_l1_loss(pred_locs,gt_locs,beta=beta,weight=weights,avg_factor=n_samples)
    return loss
        