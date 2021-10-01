import pickle
import jittor as jt 
from jittor import nn, std 
from jdet.ops.roi_align import ROIAlign
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,BOXES,LOSSES, ROI_EXTRACTORS,build_from_cfg
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.models.boxes.box_ops import delta2bbox
from jdet.ops.nms_rotated import ml_nms_rotated
from jdet.ops.nms import nms
from jdet.ops.nms_poly import nms_poly
import os
from jdet.config.constant import DOTA1_CLASSES
from numpy.lib.polynomial import poly
from jdet.models.utils.gliding_transforms import *
from jdet.models.losses import accuracy
import math

from numpy.lib.ufunclike import fix
import cv2

def print_shape(ll):
    for l in ll:
        print(l.shape)
    print("-------------------------")
    
def concat_pre(targets):
    shape = list(targets[0].shape)
    shape = shape[1:]
    shape[0] = -1
    return jt.concat([t.reshape(*shape) for t in targets])

def images_to_levels(targets,num_level_anchors):
    all_targets = []
    for target,num_level_anchor in zip(targets,num_level_anchors):
        all_targets.append(target.split(num_level_anchor,dim=0))
    
    all_targets = list(zip(*all_targets))
    targets = []
    for target in all_targets:
        targets.extend(target)
    all_targets = jt.concat(targets)
    return all_targets


@HEADS.register_module()
class GlidingHead(nn.Module):

    def __init__(self,
                 num_classes=15,
                 in_channels=256,
                 representation_dim = 1024,
                 pooler_resolution=7, 
                 pooler_scales = [1/8., 1/16., 1/32., 1/64., 1/128.],
                 pooler_sampling_ratio = 0,
                 score_thresh=0.05,
                 nms_thresh=0.7,
                 detections_per_img=100,
                 box_weights = (10., 10., 5., 5.),
                 assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                 sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 fix_coder=dict(type='GVFixCoder'),
                 ratio_coder=dict(type='GVRatioCoder'),
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 cls_loss=dict(
                     type='CrossEntropyLoss',
                     ),
                 bbox_loss=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=1.0
                     ),
                 fix_loss=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=1.0,
                     ),
                 ratio_loss=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=16.0
                     ),
     ):
        super().__init__()
        self.representation_dim = representation_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooler_resolution = pooler_resolution
        self.pooler_scales = pooler_scales
        self.pooler_sampling_ratio = pooler_sampling_ratio
        self.box_weights = box_weights
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        # TODO: Add these attr to config
        self.with_bbox = True
        self.with_shared_head = False
        self.start_bbox_type = 'hbb'
        self.end_bbox_type = 'poly'
        self.with_avg_pool = False
        self.pos_weight = -1
        self.reg_class_agnostic = False
        self.ratio_thr = 0.8
        self.max_per_img = 2000

        self.assigner = build_from_cfg(assigner, BOXES)
        self.sampler = build_from_cfg(sampler, BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.fix_coder = build_from_cfg(fix_coder, BOXES)
        self.ratio_coder = build_from_cfg(ratio_coder, BOXES)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.cls_loss = build_from_cfg(cls_loss, LOSSES)
        self.bbox_loss = build_from_cfg(bbox_loss, LOSSES)
        self.fix_loss = build_from_cfg(fix_loss, LOSSES)
        self.ratio_loss = build_from_cfg(ratio_loss, LOSSES)
        
        self._init_layers()
        self.init_weights()

    
    def _init_layers(self):
        self.roi_aligns  = [
                 ROIAlign(
                     output_size = self.pooler_resolution,
                     spatial_scale=scale,
                     sampling_ratio=self.pooler_sampling_ratio
                 ) 
                 for scale in self.pooler_scales]

        in_dim = self.pooler_resolution*self.pooler_resolution*self.in_channels
        self.fc1 = nn.Linear(in_dim, self.representation_dim)
        self.fc2 = nn.Linear(self.representation_dim, self.representation_dim)

        self.cls_score = nn.Linear(self.representation_dim, self.num_classes + 1)
        self.bbox_pred = nn.Linear(self.representation_dim, self.num_classes * 4)
        self.fix_pred = nn.Linear(self.representation_dim, self.num_classes * 4)
        self.ratio_pred = nn.Linear(self.representation_dim, self.num_classes * 1)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        nn.init.gauss_(self.cls_score.weight,std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        for l in [self.bbox_pred, self.fix_pred, self.ratio_pred]:
            nn.init.gauss_(l.weight,std=0.001)
            nn.init.constant_(l.bias, 0)

    def get_bbox_dim(self, bbox_type, with_score=False):
        
        if bbox_type == 'hbb':
            dim = 4
        elif bbox_type == 'obb':
            dim = 5
        elif bbox_type == 'poly':
            dim = 8
        else:
            raise ValueError(f"don't know {bbox_type} bbox dim")

        if with_score:
            dim += 1
        return dim

    def arb2roi(self, bbox_list, bbox_type='hbb'):

        assert bbox_type in ['hbb', 'obb', 'poly']
        bbox_dim = self.get_bbox_dim(bbox_type)

        rois_list = []
        for img_id, bboxes in enumerate(bbox_list):
            if bboxes.size(0) > 0:
                img_inds = jt.full((bboxes.size(0), 1), img_id, dtype=bboxes.dtype)
                rois = jt.concat([img_inds, bboxes[:, :bbox_dim]], dim=-1)
            else:
                rois = jt.zeros((0, bbox_dim + 1), dtype=bboxes.dtype)
            rois_list.append(rois)
        rois = jt.concat(rois_list, 0)
        return rois

    def arb2result(self, bboxes, labels, num_classes, bbox_type='hbb'):

        assert bbox_type in ['hbb', 'obb', 'poly']
        bbox_dim = self.get_bbox_dim(bbox_type, with_score=True)

        if bboxes.shape[0] == 0:
            return [jt.zeros((0, bbox_dim), dtype="float32") for i in range(num_classes)]
        else:
            return [bboxes[labels == i, :] for i in range(num_classes)]
    
    def multiclass_arb_nms(self, multi_bboxes, multi_scores, score_thr, max_num=-1, score_factors=None, bbox_type='hbb'):
        
        bbox_dim = self.get_bbox_dim(bbox_type)
        num_classes = multi_scores.size(1) - 1

        # exclude background category
        if multi_bboxes.shape[1] > bbox_dim:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, bbox_dim)
        else:
            bboxes = multi_bboxes[:, None].expand(-1, num_classes, bbox_dim)
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > self.score_thresh
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            bboxes =jt.zeros((0, bbox_dim+1), dtype=multi_bboxes.dtype)
            labels = jt.zeros((0, ), dtype="int64")
            return bboxes, labels

        dets = jt.concat([bboxes, scores.unsqueeze(1)], dim=1)

        # keep = jt.nms(dets, self.nms_thresh)

        # if max_num > 0:
        #     keep = keep[:max_num]
            
        # dets = dets[keep, :]

        return dets, labels
        
    def forward_single(self, x, sampling_results, test=False):
        
        if test:
            rois = self.arb2roi(sampling_results, bbox_type=self.start_bbox_type)
        else:
            rois = self.arb2roi([res.bboxes for res in sampling_results])
            
        ### Test begin
        # sampling_results_bboxes = []
        # for i in range(len(sampling_results)):
        #     with open(f'/mnt/disk/czh/masknet/temp/sampling_{i}.pkl', 'rb') as f:
        #         sampling_results_bboxes.append(jt.array(pickle.load(f)))

        # x = []
        # for i in range(5):
        #     with open(f'/mnt/disk/czh/masknet/temp/x_{i}.pkl', 'rb') as f:
        #         x.append(jt.array(pickle.load(f)))
            
        # x = tuple(x)    
        # rois = self.arb2roi([res for res in sampling_results_bboxes])
        ### Test end
        
        x = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

        ## Test begin
        # if test == False:
        #     with open(f'/mnt/disk/czh/masknet/temp/bbox_feats_train.pkl', 'rb') as f:
        #         x = jt.array(pickle.load(f))
        # else:
        #     with open(f'/mnt/disk/czh/masknet/temp/bbox_feats_test.pkl', 'rb') as f:
        #         x = jt.array(pickle.load(f))
        ### Test end

        if self.with_shared_head:
            x = self.shared_head(x)
        
        if self.with_avg_pool:
            x = self.avg_pool2d(x)

        x = x.view(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        fixes = self.fix_pred(x)
        fixes = fixes.sigmoid()
        ratios = self.ratio_pred(x)
        ratios = ratios.sigmoid()

        return scores, bbox_deltas, fixes, ratios, rois
    
    def loss(self, cls_score, bbox_pred, fix_pred, ratio_pred, rois, labels, label_weights, bbox_targets, bbox_weights,
             fix_targets, fix_weights, ratio_targets, ratio_weights, reduction_override=None):

        losses = dict()
        avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            losses['gliding_cls_loss'] = self.cls_loss(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            # losses['acc'] = accuracy(cls_score, labels)

        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any_():
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.astype(jt.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.astype(jt.bool),
                       labels[pos_inds.astype(jt.bool)]]

            losses['gliding_bbox_loss'] = self.bbox_loss(
                pos_bbox_pred,
                bbox_targets[pos_inds.astype(jt.bool)],
                bbox_weights[pos_inds.astype(jt.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['gliding_bbox_loss'] = bbox_pred.sum() * 0

        if pos_inds.any_():
            if self.reg_class_agnostic:
                pos_fix_pred = fix_pred.view(
                    fix_pred.size(0), 4)[pos_inds.astype(jt.bool)]
            else:
                pos_fix_pred = fix_pred.view(
                    fix_pred.size(0), -1,
                    4)[pos_inds.astype(jt.bool),
                       labels[pos_inds.astype(jt.bool)]]
            losses['gliding_fix_loss'] = self.fix_loss(
                pos_fix_pred,
                fix_targets[pos_inds.astype(jt.bool)],
                fix_weights[pos_inds.astype(jt.bool)],
                avg_factor=fix_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['gliding_fix_loss'] = fix_pred.sum() * 0

        if pos_inds.any_():
            if self.reg_class_agnostic:
                pos_ratio_pred = ratio_pred.view(
                    ratio_pred.size(0), 1)[pos_inds.astype(jt.bool)]
            else:
                pos_ratio_pred = ratio_pred.view(
                    ratio_pred.size(0), -1,
                    1)[pos_inds.astype(jt.bool),
                       labels[pos_inds.astype(jt.bool)]]
            losses['gliding_ratio_loss'] = self.ratio_loss(
                pos_ratio_pred,
                ratio_targets[pos_inds.astype(jt.bool)],
                ratio_weights[pos_inds.astype(jt.bool)],
                avg_factor=ratio_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['gliding_ratio_loss'] = ratio_pred.sum() * 0

        return losses

    def get_bboxes_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels):

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes - 1]
        labels = jt.full((num_samples,), self.num_classes, dtype="int64")
        label_weights = jt.zeros((num_samples,), dtype=pos_bboxes.dtype)
        bbox_targets = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        bbox_weights = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        fix_targets = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        fix_weights = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        ratio_targets = jt.zeros((num_samples, 1), dtype=pos_bboxes.dtype)
        ratio_weights = jt.zeros((num_samples, 1), dtype=pos_bboxes.dtype)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if self.pos_weight <= 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, bbox2type(pos_gt_bboxes, 'hbb'))
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

            pos_fix_targets = self.fix_coder.encode(bbox2type(pos_gt_bboxes, 'poly'))

            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1

            pos_ratio_targets = self.ratio_coder.encode(bbox2type(pos_gt_bboxes, 'poly'))
            
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets, fix_weights, ratio_targets, ratio_weights)
        
    def get_bboxes_targets(self, sampling_results, concat=True):

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        
        outputs = multi_apply(
            self.get_bboxes_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list)

        (labels, label_weights, bbox_targets, bbox_weights, fix_targets, fix_weights, ratio_targets, ratio_weights) = outputs

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)
            fix_targets = jt.concat(fix_targets, 0)
            fix_weights = jt.concat(fix_weights, 0)
            ratio_targets = jt.concat(ratio_targets, 0)
            ratio_weights = jt.concat(ratio_weights, 0)

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets, fix_weights, ratio_targets, ratio_weights)

    def get_bboxes(self, rois, cls_score, bbox_pred, fix_pred, ratio_pred, img_shape, scale_factor, rescale=False):
        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        
        scores = nn.softmax(cls_score, dim=1)

        bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        polys = self.fix_coder.decode(bboxes, fix_pred)

        bboxes = bboxes.view(*ratio_pred.size(), 4)
        polys = polys.view(*ratio_pred.size(), 8)
        polys[ratio_pred > self.ratio_thr] = hbb2poly(bboxes[ratio_pred > self.ratio_thr])

        if rescale:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor for _ in range(4)]
            scale_factor = jt.array(scale_factor, dtype=bboxes.dtype)
            polys /= scale_factor.repeat(2)
        polys = polys.view(polys.size(0), -1)

        det_bboxes, det_labels = self.multiclass_arb_nms(polys, scores, self.score_thresh, self.max_per_img, bbox_type='poly')
        det_labels = det_labels + 1 # output label range should be adjusted back to [1, self.class_NUm]

        return det_bboxes, det_labels

    def execute(self, x, proposal_list, targets):

        if self.is_training():

            gt_obboxes = []
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_obboxes_ignore = []

            for target in targets:
                gt_obboxes.append(target['polys'])
                gt_bboxes.append(target['hboxes'])
                gt_labels.append(target['labels'] - 1)
                gt_bboxes_ignore.append(target['hboxes_ignore'])
                gt_obboxes_ignore.append(target['polys_ignore'])

            # assign gts and sample proposals
            if self.with_bbox:
                start_bbox_type = self.start_bbox_type
                end_bbox_type = self.end_bbox_type
                target_bboxes = gt_bboxes if start_bbox_type == 'hbb' else gt_obboxes
                target_bboxes_ignore = gt_bboxes_ignore if start_bbox_type == 'hbb' else gt_obboxes_ignore

                num_imgs = len(targets)
                if target_bboxes_ignore is None:
                    target_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []

                for i in range(num_imgs):

                    assign_result = self.assigner.assign(proposal_list[i], target_bboxes[i], target_bboxes_ignore[i], gt_labels[i])

                    sampling_result = self.sampler.sample(
                        assign_result,
                        proposal_list[i],
                        target_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])

                    if start_bbox_type != end_bbox_type:
                        if gt_obboxes[i].numel() == 0:
                            # TODO: PR OBBDetection's bugs!
                            # sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeors((0, gt_obboxes[0].size(-1)))
                            sampling_result.pos_gt_bboxes = jt.zeros((0, gt_obboxes[0].size(-1)), dtype=gt_obboxes[i].dtype)
                        else:
                            sampling_result.pos_gt_bboxes = gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)
                    

            scores, bbox_deltas, fixes, ratios, rois = self.forward_single(x, sampling_results, test=False)

            ## Test begin

            # with open(f'/mnt/disk/czh/masknet/temp/cls_score.pkl', 'rb') as f:
            #     scores = jt.array(pickle.load(f))
            # with open(f'/mnt/disk/czh/masknet/temp/bbox_pred.pkl', 'rb') as f:
            #     bbox_deltas = jt.array(pickle.load(f))
            # with open(f'/mnt/disk/czh/masknet/temp/fix_pred.pkl', 'rb') as f:
            #     fixes = jt.array(pickle.load(f))
            # with open(f'/mnt/disk/czh/masknet/temp/ratio_pred.pkl', 'rb') as f:
            #     ratios = jt.array(pickle.load(f))

            ## Test end

            bbox_targets = self.get_bboxes_targets(sampling_results)

            return self.loss(scores, bbox_deltas, fixes, ratios, rois, *bbox_targets)
            
        else:
            
            result = []
            for i in range(len(targets)):

                scores, bbox_deltas, fixes, ratios, rois = self.forward_single(x, [proposal_list[i]], test=True)
                img_shape = targets[i]['img_size']
                scale_factor = targets[i]['scale_factor']
                
                det_bboxes, det_labels = self.get_bboxes(rois, scores, bbox_deltas, fixes, ratios, img_shape, scale_factor)

                poly = det_bboxes[:, :8]
                scores = det_bboxes[:, 8]
                labels = det_labels

                # visualization
                # poly_oris = polys_ori.reshape(-1, 8)

                # img_vis = cv2.imread(targets[i]["img_file"])
                # img_vis_ori = cv2.imread(targets[i]["img_file"])
                # filename = targets[i]["img_file"].split('/')[-1]
                # for j in range(poly.shape[0]):
                #     box = poly[j]
                #     draw_poly(img_vis, box.reshape(-1, 2).numpy().astype(int), color=(255,0,0))
                # for j in range(polys_ori.shape[0]):
                #     box = polys_ori[j]
                #     draw_poly(img_vis_ori, box.reshape(-1, 2).numpy().astype(int), color=(255,0,0))
                # cv2.imwrite(f'/mnt/disk/czh/gliding/visualization/test_{filename}', img_vis)
                # cv2.imwrite(f'/mnt/disk/czh/gliding/visualization/test_ori_{filename}', img_vis_ori)

                result.append((poly, scores, labels))
            
            return result

def draw_rbox(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 

def draw_box(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 

def visual_gts(targets,save_dir="./"):
    for t in targets:
        bbox = t["hboxes"].numpy()
        labels = t["labels"].numpy()
        classes = DOTA1_CLASSES
        ori_img_size = t["ori_img_size"]
        img_size = t["img_size"]
        bbox[:,0::2] *= float(ori_img_size[0]/img_size[0])
        bbox[:,1::2] *= float(ori_img_size[1]/img_size[1])
        img_f = t["img_file"]
        img = cv2.imread(img_f)
        for box,l in zip(bbox,labels):
            text = classes[l-1]
            img = draw_box(img,box,text,(255,0,0))
        cv2.imwrite(os.path.join(save_dir,"targtes.jpg"),img)

def draw_poly(img,point,color=(255,0,0),thickness=2):
    cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)


def draw_proposal(img_file,proposals):
    img = cv2.imread(img_file)
    for box in proposals:
        box = [int(x) for x in box]
        img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=(255,0,0), thickness=1)
    cv2.imwrite("proposal.jpg",img)

def draw_rboxes(img_file,boxes,scores,labels,classnames):
    img = cv2.imread(img_file)
    for box,score,label in zip(boxes,scores,labels):
        # box = rotated_box_to_poly_single(box)
        box = box.reshape(-1,2).astype(int)
        classname = classnames[label-1]
        text = f"{classname}:{score:.2}"
        draw_poly(img,box,color=(255,0,0),thickness=2)

        img = cv2.putText(img=img, text=text, org=(box[0][0],box[0][1]-5), fontFace=0, fontScale=0.5, color=(255,0,0), thickness=1)

    cv2.imwrite("test.jpg",img)

def handle_ratio_prediction(hboxes,rboxes,ratios,scores,labels):

    if rboxes.numel()==0:
        return rboxes, scores, labels

    h_idx = jt.where(ratios > 0.8)[0]
    h = hboxes[h_idx]
    hboxes_vtx = jt.concat([h[:, 0:1], h[:, 1:2], h[:, 2:3], h[:, 1:2], h[:, 2:3], h[:, 3:4], h[:, 0:1], h[:, 3:4]],dim=1)
    rboxes[h_idx] = hboxes_vtx
    # keep = nms_poly(rboxes,scores, 0.1 )

    # rboxes = rboxes[keep]
    # scores = scores[keep]
    # labels = labels[keep]

    return rboxes, scores, labels


def polygons2ratios(rbox):
    def polygon_area( corners ):
        n = len( corners ) # of corners
        area = 0.0
        for i in range( n ):
            j = ( i + 1 ) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs( area ) / 2.0
        return area
    
    max_x_ = rbox[:,  ::2].max( 1 )
    min_x_ = rbox[:,  ::2].min( 1 )
    max_y_ = rbox[:, 1::2].max( 1 )
    min_y_ = rbox[:, 1::2].min( 1 )

    rbox = rbox.view( (-1, 4, 2) )

    polygon_areas = list( map( polygon_area, rbox ) )
    polygon_areas = jt.concat( polygon_areas )

    hbox_areas = ( max_y_ - min_y_ + 1 ) * ( max_x_ - min_x_ + 1 )
    ratio_gt = polygon_areas / hbox_areas

    ratio_gt = ratio_gt.view( (-1, 1) )
    return ratio_gt

def polygons2fix(rbox):
    max_x_idx,max_x_  = rbox[:,  ::2].argmax( 1 )
    min_x_idx,min_x_  = rbox[:,  ::2].argmin( 1 )
    max_y_idx,max_y_  = rbox[:, 1::2].argmax( 1 )
    min_y_idx,min_y_  = rbox[:, 1::2].argmin( 1 )

    x_center = ( max_x_ + min_x_ ) / 2.
    y_center = ( max_y_ + min_y_ ) / 2.

    box = jt.stack( [min_x_, min_y_, max_x_, max_y_ ], dim=0 ).permute( 1, 0 )

    rbox = rbox.view( (-1, 4, 2) )

    rbox_ordered = jt.zeros_like( rbox )
    rbox_ordered[:, 0] = rbox[range(len(rbox)), min_y_idx]
    rbox_ordered[:, 1] = rbox[range(len(rbox)), max_x_idx]
    rbox_ordered[:, 2] = rbox[range(len(rbox)), max_y_idx]
    rbox_ordered[:, 3] = rbox[range(len(rbox)), min_x_idx]

    top   = rbox_ordered[:, 0, 0]
    right = rbox_ordered[:, 1, 1]
    down  = rbox_ordered[:, 2, 0]
    left  = rbox_ordered[:, 3, 1]

    """
    top = jt.min( jt.max( top, box[:, 0] ), box[:, 2] )
    right = jt.min( jt.max( right, box[:, 1] ), box[:, 3] )
    down = jt.min( jt.max( down, box[:, 0] ), box[:, 2] )
    left = jt.min( jt.max( left, box[:, 1] ), box[:, 3] )
    """

    top_gt = (top - box[:, 0]) / (box[:, 2] - box[:, 0])
    right_gt = (right - box[:, 1]) / (box[:, 3] - box[:, 1])
    down_gt = (box[:, 2] - down) / (box[:, 2] - box[:, 0])
    left_gt = (box[:, 3] - left) / (box[:, 3] - box[:, 1])

    hori_box_mask = ((rbox_ordered[:,0,1] - rbox_ordered[:,1,1]) == 0) + ((rbox_ordered[:,1,0] - rbox_ordered[:,2,0]) == 0)

    fix_gt = jt.stack( [top_gt, right_gt, down_gt, left_gt] ).permute( 1, 0 )
    fix_gt = fix_gt.view( (-1, 4) )
    fix_gt[hori_box_mask, :] = 1
    return fix_gt

def fix2polygons(box, alphas):
    pred_top = (box[:, 2::4] - box[:, 0::4]) * alphas[:, 0::4] + box[:, 0::4]
    pred_right = (box[:, 3::4] - box[:, 1::4]) * alphas[:, 1::4] + box[:, 1::4]
    pred_down = (box[:, 0::4] - box[:, 2::4]) * alphas[:, 2::4] + box[:, 2::4]
    pred_left = (box[:, 1::4] - box[:, 3::4]) * alphas[:, 3::4] + box[:, 3::4]

    pred_rbox = jt.zeros( (box.shape[0], box.shape[1] * 2) )
    pred_rbox[:, 0::8] = pred_top
    pred_rbox[:, 1::8] = box[:, 1::4]
    pred_rbox[:, 2::8] = box[:, 2::4]
    pred_rbox[:, 3::8] = pred_right
    pred_rbox[:, 4::8] = pred_down
    pred_rbox[:, 5::8] = box[:, 3::4]
    pred_rbox[:, 6::8] = box[:, 0::4]
    pred_rbox[:, 7::8] = pred_left
    return pred_rbox
