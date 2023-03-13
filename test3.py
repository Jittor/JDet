import argparse
import jittor as jt
jt.flags.use_cuda_managed_allocator=1
jt.flags.use_cuda=1
from jdet.runner import Runner 
from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, BOXES
import pickle
from jdet.ops.reppoints_convex_iou import reppoints_convex_giou, reppoints_convex_iou

def main1():
    value_dict = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/result_dict.pkl", "rb"))
    points = value_dict['points']
    gt_rbboxes = value_dict['gt_rbboxes']
    overlaps = value_dict['overlaps']
    jt_overlaps = reppoints_convex_iou(jt.array(points), jt.array(gt_rbboxes))
    print(jt_overlaps.shape)
    print(gt_rbboxes.shape, points.shape, overlaps.shape)
    print((jt_overlaps - overlaps).abs().max().item(), jt.array(overlaps).abs().max().item())
    print(jt_overlaps.abs().mean().item(), jt.array(overlaps).abs().mean().item())
    return

def main2():
    value_dict = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/grad_dict.pkl", "rb"))
    pred = value_dict['pred']
    target = value_dict['target']
    convex_gious = value_dict['convex_gious']
    grad = value_dict['grad']
    print(pred.shape, target.shape)
    jt_giou, jt_grad = reppoints_convex_giou(jt.array(pred), jt.array(target))
    print(convex_gious.shape, jt_giou.shape)
    print(grad.shape, jt_grad.shape)
    print((jt_giou - convex_gious).abs().max().item(), jt.array(convex_gious).abs().max().item())
    print((jt_grad - grad).abs().max().item(), jt.array(grad).abs().max().item())
    print(jt_giou.abs().mean().item(), jt.array(convex_gious).abs().mean().item())
    print(jt_grad.abs().mean().item(), jt.array(grad).abs().mean().item())


def main3():
    value_dict = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/value_dict", "rb"))
    pred = value_dict['pred'][:2]
    target = value_dict['target'][:2]
    print(pred)
    print(target)
    convex_gious, grad = reppoints_convex_giou(jt.array(pred), jt.array(target))
    print(grad)

    loss = 1 - convex_gious
    reduction = 'mean'
    avg_factor = None

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor

    
    unvaild_inds = jt.nonzero(jt.any(grad > 1, dim=1))[:, 0]
    grad[unvaild_inds] = 1e-6


if __name__ == "__main__":
    # main1()
    # main2()
    main3()
