import torch
import jittor as jt
from jdet.utils.registry import HEADS, build_from_cfg
import numpy as np
rpn_head=dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_scales=[8],
    anchor_ratios=[0.5, 1.0, 2.0],
    anchor_strides=[4, 8, 16, 32, 64],
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0))

train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr = 0.05, nms = dict(type='nms', iou_thr=0.5), max_per_img = 2000)
)


rpn_head=dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_scales=[8],
    anchor_ratios=[0.5, 1.0, 2.0],
    anchor_strides=[4, 8, 16, 32, 64],
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0))
'''
rpn_model = builder.build_head(rpn_head)
rpn_model.load_state_dict(torch.load('fake_rpn.pth'))
rpn_model.to('cuda')

anchor_head_input = torch.load('anchor_head_input.pt')
anchor_head_output = rpn_model(anchor_head_input)
torch.save(anchor_head_output, 'anchor_head_std.pt')

loss_input = torch.load('loss_input.pt')
loss_output = rpn_model.loss(cls_scores=loss_input['cls_scores'],
                             bbox_preds=loss_input['bbox_preds'],
                             gt_bboxes=loss_input['gt_bboxes'],
                             img_metas=loss_input['img_metas'],
                             cfg = loss_input['cfg'])
torch.save(loss_output, 'loss_std.pt')

get_bboxes_input = torch.load('get_bboxes_input.pt')
get_bboxes_output = rpn_model.get_bboxes(
                                         cls_scores = get_bboxes_input['cls_scores'],
                                         bbox_preds = get_bboxes_input['bbox_preds'],
                                         img_metas = get_bboxes_input['img_metas'],
                                         cfg = get_bboxes_input['cfg'])
torch.save(get_bboxes_output, 'get_bboxes_std.pt')
'''
np.random.seed(514)
rpn_model = build_from_cfg(rpn_head, HEADS)
rpn_model.load('fake_rpn.pth')

loss_input = torch.load('loss_input.pt')
loss_cls_scores = []
for cls_score in loss_input['cls_scores']:
    loss_cls_scores.append(jt.array(cls_score.cpu().detach().numpy()))
loss_bbox_preds = []
for bbox_pred in loss_input['bbox_preds']:
    loss_bbox_preds.append(jt.array(bbox_pred.cpu().detach().numpy()))
loss_gt_bboxes = []
for gt_bbox in loss_input['gt_bboxes']:
    loss_gt_bboxes.append(jt.array(gt_bbox.cpu().detach().numpy()))
loss_img_metas = loss_input['img_metas']
loss_cfg = loss_input['cfg']

get_bboxes_input = torch.load('get_bboxes_input.pt')
bboxes_cls_scores = []
for cls_score in get_bboxes_input['cls_scores']:
    bboxes_cls_scores.append(jt.array(cls_score.cpu().detach().numpy()))
bboxes_bbox_preds = []
for bbox_pred in get_bboxes_input['bbox_preds']:
    bboxes_bbox_preds.append(jt.array(bbox_pred.cpu().detach().numpy()))
bboxes_img_metas = get_bboxes_input['img_metas']
bboxes_cfg = get_bboxes_input['cfg']

anchor_head_input = torch.load('anchor_head_input.pt')
anchor_feats = tuple()
for feat in anchor_head_input:
    anchor_feats += (jt.array(feat.cpu().detach().numpy()), )
loss_output = rpn_model.loss(cls_scores=loss_cls_scores,
                             bbox_preds=loss_bbox_preds,
                             gt_bboxes=loss_gt_bboxes,
                             img_metas=loss_img_metas,
                             cfg=loss_cfg)
print(loss_output)
get_bboxes_output = rpn_model.get_bboxes(
                                         cls_scores = bboxes_cls_scores,
                                         bbox_preds = bboxes_bbox_preds,
                                         img_metas = bboxes_img_metas,
                                         cfg = bboxes_cfg)
get_bboxes_std = torch.load('get_bboxes_std.pt')