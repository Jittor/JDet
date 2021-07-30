import jittor as jt 
from jittor import nn, std 
from jdet.ops.roi_align import ROIAlign
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,BOXES,LOSSES,build_from_cfg
from jdet.models.boxes.box_ops import delta2bbox
from jdet.ops.nms_rotated import ml_nms_rotated
from jdet.ops.nms import nms
from jdet.ops.nms_poly import nms_poly
import os
from jdet.config.constant import DOTA1_CLASSES
# from jdet.utils.visualization import draw_rboxes
import math
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
                 num_classes,
                 in_channels,
                 representation_dim = 1024,
                 pooler_resolution =  7, 
                 pooler_scales = [1/8., 1/16., 1/32., 1/64., 1/128.],
                 pooler_sampling_ratio = 0,
                 score_thresh=0.05,
                 nms_thresh=0.5,
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
                 cls_loss=dict(
                      type='CrossEntropyLoss',
                      ),
                 reg_loss=dict(
                      type='SmoothL1Loss', 
                      beta=1.0 / 9.0, 
                      loss_weight=1.0),
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

        self.assigner = build_from_cfg(assigner,BOXES)
        self.sampler = build_from_cfg(sampler,BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder,BOXES)
        self.cls_loss = build_from_cfg(cls_loss,LOSSES)
        self.reg_loss = build_from_cfg(reg_loss,LOSSES)

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
        self.fc1 = nn.Linear(in_dim,self.representation_dim)
        self.fc2 = nn.Linear(self.representation_dim,self.representation_dim)

        self.cls_score = nn.Linear(self.representation_dim, self.num_classes)
        self.bbox_pred = nn.Linear(self.representation_dim, self.num_classes * 4)
        self.fix_pred = nn.Linear(self.representation_dim, self.num_classes * 4)
        self.ratio_pred = nn.Linear(self.representation_dim, self.num_classes * 1)
        

    def init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, a=1)
        nn.init.constant_(self.fc2.bias, 0)

        for l in [self.cls_score, self.bbox_pred, self.fix_pred, self.ratio_pred]:
            nn.init.gauss_(l.weight,std=0.001)
            nn.init.constant_(l.bias, 0)


    def forward_single(self,x,roi_align,one_level_proposals):
        indices = jt.concat([jt.ones((proposal.shape[0],))*idx
                               for idx,proposal in enumerate(one_level_proposals)])
        proposals = jt.concat(one_level_proposals)
        rois = jt.concat([indices.unsqueeze(-1),proposals],dim=1)
        assert rois.shape[1]==5
        x = roi_align(x,rois)
        x = x.view(x.shape[0],-1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        fixes = self.fix_pred(x)
        fixes = fixes.sigmoid()
        ratios = self.ratio_pred(x)
        ratios = ratios.sigmoid()

        return scores, bbox_deltas, fixes, ratios,indices

    
    def _get_targets_single(self,mlvl_anchors,target):
        """Compute regression and classification targets for anchors in a
        single image.
        """
        # w,h
        gt_bboxes = target["hboxes"]
        gt_labels = target["labels"]
        gt_polys = target["polys"]
        gt_fixes = polygons2fix(gt_polys)
        gt_ratios = polygons2ratios(gt_polys)
        anchors = jt.concat(mlvl_anchors)

        # assign gt and sample anchors
        assign_result = self.assigner.assign(anchors, gt_bboxes,gt_labels=gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,gt_bboxes,gt_labels=gt_labels)

        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds


        num_valid_anchors = anchors.shape[0]
        bbox_targets = jt.zeros_like(anchors)
        bbox_weights = jt.zeros_like(anchors)
        fix_targets = jt.zeros_like(anchors)
        fix_weights = jt.zeros_like(anchors)
        ratios_targets = jt.zeros((num_valid_anchors,1))
        ratios_weights = jt.zeros((num_valid_anchors,1))
        # 1 is background label
        labels = jt.zeros((num_valid_anchors, )).int()
        label_weights = jt.ones((num_valid_anchors,)).float()

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # which is box delta
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            fix_targets[pos_inds,:]=gt_fixes[pos_assigned_gt_inds,:]
            fix_weights[pos_inds,:]=1.0
            ratios_targets[pos_inds,:]=gt_ratios[pos_assigned_gt_inds,:]
            ratios_weights[pos_inds,:]=1.0
            labels[pos_inds] = sampling_result.pos_gt_labels
        
        return (labels, label_weights, bbox_targets, bbox_weights,fix_targets, fix_weights,ratios_targets,ratios_weights,pos_inds,neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    targets):
        """Compute regression and classification targets for anchors in
        multiple images.
        """

        # anchor number of multi levels
        num_level_anchors = [ [anchors.size(0) for anchors in anchor_l ] for anchor_l in anchor_list]


        # compute targets for each image
        (all_labels, all_label_weights, all_bbox_targets, 
           all_bbox_weights, all_fix_targets, all_fix_weights,all_ratios_targets,all_ratios_weights,
           pos_inds_list, neg_inds_list, sampling_results_list) = multi_apply(self._get_targets_single,anchor_list,targets)

        
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels


        labels_list = images_to_levels(all_labels, num_level_anchors)

        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        
        fix_targets_list = images_to_levels(all_fix_targets,
                                             num_level_anchors)
        fix_weights_list = images_to_levels(all_fix_weights,
                                             num_level_anchors)

        ratios_targets_list = images_to_levels(all_ratios_targets,
                                             num_level_anchors)
        ratios_weights_list = images_to_levels(all_ratios_weights,
                                             num_level_anchors)

        return labels_list, label_weights_list, bbox_targets_list,bbox_weights_list,fix_targets_list,fix_weights_list,ratios_targets_list,ratios_weights_list, num_total_pos, num_total_neg

        

    def loss(self, scores, bbox_deltas, fixes, ratios,indices,proposals,targets):
        scores = jt.concat(scores)
        bbox_deltas = jt.concat(bbox_deltas)
        fixes = jt.concat(fixes)
        ratios = jt.concat(ratios)
        indices = jt.concat(indices)

        (labels, label_weights,
         bbox_targets,bbox_weights,
         fix_targets,fix_weights,
         ratios_targets,ratios_weights, 
         num_total_pos, num_total_neg) = self.get_targets(proposals,targets)

        num_total_samples =  num_total_pos + num_total_neg 

        # labels = concat_pre(labels_list)
        # label_weights = concat_pre(label_weights_list)
        # bbox_targets= concat_pre(bbox_targets_list)
        # bbox_weights= concat_pre(bbox_weights_list)
        # fix_targets = concat_pre(fix_targets_list)
        # fix_weights= concat_pre(fix_weights_list)
        # ratios_targets = concat_pre(ratios_targets_list)
        # ratios_weights = concat_pre(ratios_weights_list)

        cls_loss = self.cls_loss(scores,labels)
        
        bbox_deltas = bbox_deltas.reshape(-1,self.num_classes,4)
        bbox_deltas = bbox_deltas[range(labels.shape[0]),labels]
        bbox_loss = self.reg_loss(bbox_deltas,bbox_targets,bbox_weights,avg_factor=num_total_samples)
        
        fixes = fixes.reshape(-1,self.num_classes,4)
        fixes = fixes[range(labels.shape[0]),labels]
        assert fixes.max().item()<=1 and fixes.min().item()>=0
        fix_loss = self.reg_loss(fixes,fix_targets,fix_weights,avg_factor=num_total_samples)
        
        ratios = ratios.reshape(-1,self.num_classes,1)
        ratios = ratios[range(labels.shape[0]),labels]
        assert ratios.max().item()<=1 and ratios.min().item()>=0
        ratio_loss = self.reg_loss(ratios,ratios_targets,ratios_weights,avg_factor=num_total_samples)

        return dict(
                 gliding_cls_loss=cls_loss,
                 gliding_bbox_loss=bbox_loss,
                 gliding_fix_loss=fix_loss,
                 gliding_ratio_loss=ratio_loss,
             )


    def get_bboxes_single(self,class_logit, box_reg, fix_reg, ratio_reg,proposal,target):

        assert len(class_logit)==len(proposal),f"class_logits {len(class_logit)} != proposals {len(proposal)}"

        
        # # TODO maybe it's wrong
        labels = jt.arange(self.num_classes).reshape(1,-1)
        labels = labels.repeat(proposal.shape[0],1).reshape(-1)
        proposal = proposal.unsqueeze(1)
        proposal = proposal.repeat(1,self.num_classes,1).reshape(-1,4)

        # get_bboxes for single images
        scores = nn.softmax(class_logit, -1)
        box_reg = box_reg.reshape(-1,4)
        assert proposal.shape[0]==box_reg.shape[0]


        boxes = self.bbox_coder.decode(proposal,box_reg)
        polygons = fix2polygons(boxes,fix_reg.reshape(-1,4))
        boxes = boxes.reshape(-1,self.num_classes,4)
        polygons = polygons.reshape(-1,self.num_classes,8)

        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, self.num_classes):
            inds = jt.where(inds_all[:, j])[0]
            scores_j = scores[inds, j]
            ratios_j = ratio_reg[inds, j]
            # alphas_j = fix_reg[inds,j]
            boxes_j = boxes[inds, j]
            rboxes_j = polygons[inds, j]

            keep = nms(boxes_j,scores_j,self.nms_thresh)

            scores_j = scores_j[keep]
            ratios_j = ratios_j[keep]
            # alphas_j = alphas_j[keep]
            boxes_j = boxes_j[keep]
            rboxes_j = rboxes_j[keep]
            labels_j = jt.ones((scores_j.shape[0],)).int()*j 

            result.append((boxes_j,rboxes_j,ratios_j,scores_j,labels_j))

        
        result = [jt.concat(r) for r in zip(*result)]
        number_of_detections = len(result[0])

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            scores = result[3]

            keep,scores = jt.argsort(scores,descending=True)
            keep = keep[:self.detections_per_img]
            result = [r[keep] for r in result]
        polys,scores,labels = handle_ratio_prediction(*result)
        # print(polys.shape,scores.shape,labels.shape)
        return polys,scores,labels
        
    def get_bboxes(self, scores, bbox_deltas, fixes, ratios,indices,proposals,targets):
        
        scores = jt.concat(scores)
        bbox_deltas = jt.concat(bbox_deltas)
        fixes = jt.concat(fixes)
        ratios = jt.concat(ratios)
        indices = jt.concat(indices)
        
        results = []
        for img_id,(proposal,target) in enumerate(zip(proposals,targets)):
            index = jt.where(indices==img_id)[0]
            single_image_scores = scores[index]
            single_image_bbox_deltas = bbox_deltas[index]
            single_image_fixes = fixes[index]
            single_image_ratios = ratios[index]
            proposal = jt.concat(proposal)
            result = self.get_bboxes_single(single_image_scores,single_image_bbox_deltas,single_image_fixes,single_image_ratios,proposal,target)
            results.append(result)

        return results

    def execute(self,features,proposals,targets):
        """
        proposals: list[list[Tensor]]=[[image0_level0_proposal,image0_level1_proposal],[image1_level0_proposal,image1_level1_proposal]]
        """
        # src_proposals = proposals 
        # proposals = [[pp[0] for pp in p] for p in proposals]

        level_proposals = list(zip(*proposals))
        outs = multi_apply(self.forward_single,features,self.roi_aligns,level_proposals)
        # if hasattr(self,"test_iter"):
        #     self.test_iter+=1
        # else:
        #     self.test_iter = 1
        
        # if self.test_iter>2500:
        #     pp = jt.concat([p[0] for p in src_proposals[0]])
        #     scores = jt.concat([p[1] for p in src_proposals[0]])
        #     keep = scores>0.9
        #     pp = pp[keep]
        #     img_file = targets[0]["img_file"]
        #     draw_proposal(img_file,pp.numpy())
        #     visual_gts(targets)
        #     result = self.get_bboxes(*outs,proposals,targets)
        #     for r,t in zip(result,targets):
        #         img_file = t['img_file']
        #         polys,scores,labels = r 
        #         draw_rboxes(img_file,polys.numpy(),scores.numpy(),labels.numpy(),DOTA1_CLASSES)
        #     exit()
                
        if self.is_training():
            return self.loss(*outs,proposals,targets)
        else:
            return self.get_bboxes(*outs,proposals,targets)

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
    keep = nms_poly(rboxes,scores, 0.1 )

    rboxes = rboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

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
    top = torch.min( torch.max( top, box[:, 0] ), box[:, 2] )
    right = torch.min( torch.max( right, box[:, 1] ), box[:, 3] )
    down = torch.min( torch.max( down, box[:, 0] ), box[:, 2] )
    left = torch.min( torch.max( left, box[:, 1] ), box[:, 3] )
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
