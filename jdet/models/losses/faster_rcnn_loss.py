import jittor as jt 

# from https://github.com/chenyuntc/simple-faster-rcnn-pytorch/
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

# from https://github.com/chenyuntc/simple-faster-rcnn-pytorch/
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = jt.zeros(gt_loc.shape)
    # Localization loss is calculated only for positive rois.
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    # ignore gt_label==-1 for rpn_loss
    loc_loss /= ((gt_label >= 0).sum().float()) 
    return loc_loss