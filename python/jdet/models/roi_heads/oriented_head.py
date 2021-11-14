import pickle
import jittor as jt 
from jittor import nn
from jdet.data.devkits.result_merge import py_cpu_nms_poly_fast
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,BOXES,LOSSES, ROI_EXTRACTORS,build_from_cfg

from jdet.ops.bbox_transforms import *
from jdet.models.utils.modules import ConvModule

from jittor.misc import _pair

@HEADS.register_module()
class OrientedHead(nn.Module):

    def __init__(self,
                 num_classes=15,
                 in_channels=256,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 fc_out_channels=1024,
                 conv_out_channels=256,
                 score_thresh=0.05,
                 assigner=dict(
                     type='MaxIoUAssigner',
                     pos_iou_thr=0.5,
                     neg_iou_thr=0.5,
                     min_pos_iou=0.5,
                     ignore_iof_thr=-1,
                     match_low_quality=False,
                     iou_calculator=dict(type='BboxOverlaps2D')),
                 sampler=dict(
                     type='RandomSampler',
                     num=512,
                     pos_fraction=0.25,
                     neg_pos_ub=-1,
                     add_gt_as_proposals=True),
                 bbox_coder=dict(
                     type='OrientedDeltaXYWHTCoder',
                     target_means=[0., 0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                 bbox_roi_extractor=dict(
                     type='OrientedSingleRoIExtractor',
                     roi_layer=dict(type='ROIAlignRotated', out_size=7, sample_num=2),
                     out_channels=256,
                     extend_factor=(1.4, 1.2),
                     featmap_strides=[4, 8, 16, 32]),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                    ),
                 loss_bbox=dict(
                     type='SmoothL1Loss', 
                     beta=1.0, 
                     loss_weight=1.0
                     ),
                 with_bbox=True,
                 with_shared_head=False,
                 with_cls=True,
                 with_reg=True,
                 with_avg_pool=False,
                 start_bbox_type='obb',
                 end_bbox_type='obb',
                 reg_dim=None,
                 pos_weight=-1,
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 
     ):
        super().__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.with_bbox = with_bbox
        self.with_shared_head = with_shared_head
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False
        self.pos_weight = pos_weight
        self.score_thresh = score_thresh
        
        # TODO Add to config
        roi_feat_size = 7
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]

        self.start_bbox_type = start_bbox_type
        self.end_bbox_type = end_bbox_type
        assert self.start_bbox_type in ['hbb', 'obb', 'poly']
        assert self.end_bbox_type in ['hbb', 'obb', 'poly']
        self.reg_dim = get_bbox_dim(self.end_bbox_type) \
                if reg_dim is None else reg_dim

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
            
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        if self.with_reg:
            out_dim_reg = self.reg_dim if reg_class_agnostic else self.reg_dim * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)

        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels

        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.assigner = build_from_cfg(assigner, BOXES)
        self.sampler = build_from_cfg(sampler, BOXES)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        
        self._init_layers()
        self.init_weights()

    def _add_conv_fc_branch(self, num_branch_convs,  num_branch_fcs, in_channels, is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels

        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=None))
            last_layer_dim = self.conv_out_channels

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append( nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels

        return branch_convs, branch_fcs, last_layer_dim

    def _init_layers(self):

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = self.reg_dim if self.reg_class_agnostic else \
                    self.reg_dim * self.num_classes
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def init_weights(self):

        if self.with_cls:
            nn.init.gauss_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.gauss_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def arb2roi(self, bbox_list, bbox_type='hbb'):

        assert bbox_type in ['hbb', 'obb', 'poly']
        bbox_dim = get_bbox_dim(bbox_type)

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
    
    def get_results(self, multi_bboxes, multi_scores, score_factors=None, bbox_type='hbb'):
        
        bbox_dim = get_bbox_dim(bbox_type)
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
            bboxes = jt.zeros((0, 9), dtype=multi_bboxes.dtype)
            labels = jt.zeros((0, ), dtype="int64")
            return bboxes, labels

        # hbboxes = obb2hbb(bboxes)
        # max_coordinate = hbboxes.max() - hbboxes.min()
        # offsets = labels.astype(bboxes.dtype) * (max_coordinate + 1)
        # bboxes_for_nms = bboxes.clone()
        # bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]

        # print("bbox before nms")
        # print(bboxes.shape)
        # print(bboxes)

        # print("test change")
        # print(bboxes)
        # print(poly2obb(obb2poly(bboxes)))

        dets = jt.concat([obb2poly(bboxes), scores.unsqueeze(1)], dim=1)

        # keep = py_cpu_nms_poly_fast(np.array(dets), 0.1)
        # dets = dets[keep]

        # print("bbox after nms")
        # print(dets.shape)
        # print(dets)
        
        # dets = jt.concat([bboxes, scores.unsqueeze(1)], dim=1)
        return dets, labels
        
    def forward_single(self, x, sampling_results, test=False):

        if test:
            rois = self.arb2roi(sampling_results, bbox_type=self.start_bbox_type)
        else:
            rois = self.arb2roi([res.bboxes for res in sampling_results], bbox_type=self.start_bbox_type)

        ### Test begin
        # xx = []
        # for i in range(len(x)):
        #     with open(f'/mnt/disk/czh/masknet/temp/bbox_feats_{i}.pkl', 'rb') as f:
        #         xx.append(jt.array(pickle.load(f)))
        # x = tuple(xx)
        ### Test end

        x = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = nn.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = nn.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = nn.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, rois
    
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):

        losses = dict()
        if cls_score is not None:
            avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)

        if bbox_pred is not None:

            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
                 
            ### Test begin
            # with open(f'/mnt/disk/czh/masknet/temp/pos_inds.pkl', 'rb') as f:
            #     pos_inds = jt.array(pickle.load(f))
            # with open(f'/mnt/disk/czh/masknet/temp/bbox_pred.pkl', 'rb') as f:
            #     bbox_pred = jt.array(pickle.load(f))
            # with open(f'/mnt/disk/czh/masknet/temp/rois.pkl', 'rb') as f:
            #     rois = jt.array(pickle.load(f))
            ## Test end

            # do not perform bounding box regression for BG anymore.
            if pos_inds.any_():
                
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), self.reg_dim)[pos_inds.astype(jt.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, self.reg_dim)[pos_inds.astype(jt.bool), labels[pos_inds.astype(jt.bool)]]
                
                losses['orcnn_bbox_loss'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.astype(jt.bool)],
                    bbox_weights[pos_inds.astype(jt.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['orcnn_bbox_loss'] = bbox_pred.sum() * 0

        # print("loss situation 2")
        # print(losses)

        return losses

    def get_bboxes_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels):

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes - 1]
        labels = jt.full((num_samples,), self.num_classes).long()
        label_weights = jt.zeros((num_samples,), dtype=pos_bboxes.dtype)
        bbox_targets = jt.zeros((num_samples, self.reg_dim), dtype=pos_bboxes.dtype)
        bbox_weights = jt.zeros((num_samples, self.reg_dim), dtype=pos_bboxes.dtype)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if self.pos_weight <= 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights)
        
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

        (labels, label_weights, bbox_targets, bbox_weights) = outputs

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)

        return (labels, label_weights, bbox_targets, bbox_weights)

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):
        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        
        scores = nn.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            assert self.start_bbox_type == self.end_bbox_type
            bboxes = rois[:, 1:].clone()

        if rescale:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor for _ in range(4)]
            scale_factor = jt.array(scale_factor, dtype=bboxes.dtype)

            bboxes = bboxes.view(bboxes.size(0), -1, get_bbox_dim(self.end_bbox_type))
            if self.end_bbox_type == 'hbb':
                bboxes /= scale_factor
            elif self.end_bbox_type == 'obb':
                bboxes[..., :4] = bboxes[..., :4] / scale_factor
            elif self.end_bbox_type == 'poly':
                bboxes /= scale_factor.repeat(2)
            bboxes = bboxes.view(bboxes.size(0), -1)

        # print(bboxes.shape)

        det_bboxes, det_labels = self.get_results(bboxes, scores, bbox_type=self.end_bbox_type)
        
        # print(det_bboxes.shape)

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
                gt_obboxes.append(target['rboxes'])
                gt_bboxes.append(target['hboxes'])
                gt_labels.append(target['labels'] - 1)
                gt_bboxes_ignore.append(target['hboxes_ignore'])
                gt_obboxes_ignore.append(target['rboxes_ignore'])

            ### Test Start
            
            # import pickle
            # for i in range(2):
            #     with open(f'/mnt/disk/czh/masknet/temp/ohead_proposal_{i}.pkl', 'rb') as f:
            #         proposal_list[i] = jt.array(pickle.load(f))
            #     with open(f'/mnt/disk/czh/masknet/temp/ohead_gt_obboxes_{i}.pkl', 'rb') as f:
            #         gt_obboxes.append(jt.array(pickle.load(f)))
            #     with open(f'/mnt/disk/czh/masknet/temp/ohead_gt_bboxes_{i}.pkl', 'rb') as f:
            #         gt_bboxes.append(jt.array(pickle.load(f)))
            #     with open(f'/mnt/disk/czh/masknet/temp/ohead_gt_labels_{i}.pkl', 'rb') as f:
            #         gt_labels.append(jt.array(pickle.load(f)))
            # gt_bboxes_ignore = None
            # gt_obboxes_ignore = None
            
            ### Test End

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
                            sampling_result.pos_gt_bboxes = jt.zeros((0, gt_obboxes[0].size(-1)), dtype=gt_obboxes[i].dtype)
                        else:
                            sampling_result.pos_gt_bboxes = gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

                    ### Test begin

                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_pos_is_gt_{i}.pkl', 'rb') as f:
                    #     sampling_result.pos_is_gt = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_num_gts_{i}.pkl', 'rb') as f:
                    #     sampling_result.num_gts = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_pos_gt_bboxes_{i}.pkl', 'rb') as f:
                    #     sampling_result.pos_gt_bboxes = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_pos_gt_labels_{i}.pkl', 'rb') as f:
                    #     sampling_result.pos_gt_labels = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_pos_assigned_gt_inds_{i}.pkl', 'rb') as f:
                    #     sampling_result.pos_assigned_gt_inds = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_pos_bboxes_{i}.pkl', 'rb') as f:
                    #     sampling_result.pos_bboxes = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_neg_bboxes_{i}.pkl', 'rb') as f:
                    #     sampling_result.neg_bboxes = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_pos_inds_{i}.pkl', 'rb') as f:
                    #     sampling_result.pos_inds = jt.array(pickle.load(f))
                    # with open(f'/mnt/disk/czh/masknet/temp/ohead_sr_neg_inds_{i}.pkl', 'rb') as f:
                    #     sampling_result.neg_inds = jt.array(pickle.load(f))

                    ### Test end

                    sampling_results.append(sampling_result)
                    

            scores, bbox_deltas, rois = self.forward_single(x, sampling_results, test=False)

            bbox_targets = self.get_bboxes_targets(sampling_results)
            
            return self.loss(scores, bbox_deltas, rois, *bbox_targets)
            
        else:
            
            result = []
            for i in range(len(targets)):

                scores, bbox_deltas, rois = self.forward_single(x, [proposal_list[i]], test=True)
                img_shape = targets[i]['img_size']
                scale_factor = targets[i]['scale_factor']

                ### Test begin
                
                # with open(f'/mnt/disk/czh/masknet/temp/cls_score.pkl', 'rb') as f:
                #     scores = jt.array(pickle.load(f))
                # with open(f'/mnt/disk/czh/masknet/temp/bbox_pred.pkl', 'rb') as f:
                #     bbox_deltas = jt.array(pickle.load(f))
                # with open(f'/mnt/disk/czh/masknet/temp/rois.pkl', 'rb') as f:
                #     rois = jt.array(pickle.load(f))
                
                ### Test end
                
                det_bboxes, det_labels = self.get_bboxes(rois, scores, bbox_deltas, img_shape, scale_factor, rescale=True)

                poly = det_bboxes[:, :8]
                scores = det_bboxes[:, 8]
                labels = det_labels

                # visualization
                # img_vis = cv2.imread(targets[i]["img_file"])
                # filename = targets[i]["img_file"].split('/')[-1]
                # for j in range(poly.shape[0]):
                #     box = poly[j]
                #     draw_poly(img_vis, box.reshape(-1, 2).numpy().astype(int), color=(255,0,0))
                # cv2.imwrite(f'/mnt/disk/czh/orcnn/visualization/test_{filename}', img_vis)

                # result.append((det_bboxes, det_labels))
                result.append((poly, scores, labels))

            # import sys
            # sys.exit()
            
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