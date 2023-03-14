import jittor as jt 
from jdet.utils.registry import BOXES,build_from_cfg
from jdet.models.boxes.box_ops import points_in_rotated_boxes

import numpy as np
def deleteme(a, b, size = 10):
    if a is None and b is None:
        return
    if isinstance(a, list) and isinstance(b, list):
        for a1, b1 in zip(a, b):
            print('-' * size)
            deleteme(a1, b1, size + 10)
            print('-' * size)
    elif isinstance(a, jt.Var) and isinstance(b, np.ndarray):
        print((a - b).abs().max().item())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        print(np.max(np.abs(a - b)))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        print("number diff:", a - b)
    else:
        print(type(a))
        print(type(b))
        raise NotImplementedError
def transpose_to(a, b):
    if a is None:
        return None
    if isinstance(a, list) and isinstance(b, list):
        rlist = []
        for a1, b1 in zip(a, b):
            rlist.append(transpose_to(a1, b1))
        return rlist
    elif isinstance(a, dict) and isinstance(b, dict):
        rdict = []
        for k in b.keys():
            rdict[k] = transpose_to(a[k], b[k])
        return rdict
    elif isinstance(a, np.ndarray) and isinstance(b, jt.Var):
        return jt.array(a)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a
    elif isinstance(a, tuple) and isinstance(b, tuple):
        rlist = [transpose_to(a1, b1) for a1, b1 in zip(a, b)]
        return tuple(rlist)
    elif isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
        assert(type(a) == type(b))
        return a
    else:
        print(type(a))
        print(type(b))
        raise NotImplementedError


class AssignResult:
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        self_inds = jt.arange(1, len(gt_labels) + 1).int()
        self.gt_inds = jt.contrib.concat([self_inds, self.gt_inds])
        self.max_overlaps = jt.contrib.concat([jt.ones((self.num_gts,),dtype=self.max_overlaps.dtype), self.max_overlaps])
        if self.labels is not None:
            self.labels = jt.contrib.concat([gt_labels, self.labels])

@BOXES.register_module()
class MaxIoUAssigner:
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 assigned_labels_filled=0,
                 iou_calculator=dict(type='BboxOverlaps2D')):

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.match_low_quality = match_low_quality
        self.assigned_labels_filled = assigned_labels_filled
        self.iou_calculator = build_from_cfg(iou_calculator,BOXES)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')

        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        
        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps= ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(  
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = jt.full((num_bboxes,), -1).int()

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        argmax_overlaps,max_overlaps = overlaps.argmax(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_argmax_overlaps,gt_max_overlaps = overlaps.argmax(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        if self.match_low_quality:
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
                if i % 100 == 99:
                    jt.sync_all()

        if gt_labels is not None:
            assigned_labels = jt.full((num_bboxes, ), self.assigned_labels_filled, dtype=assigned_gt_inds.dtype)
            pos_inds = jt.nonzero(assigned_gt_inds > 0).squeeze(-1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

@BOXES.register_module()
class MaxIoUAssignerRbbox(MaxIoUAssigner):
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        super(MaxIoUAssignerRbbox, self).__init__(pos_iou_thr=pos_iou_thr,
                 neg_iou_thr=neg_iou_thr,
                 min_pos_iou=min_pos_iou,
                 gt_max_assign_all=gt_max_assign_all,
                 ignore_iof_thr=ignore_iof_thr,
                 ignore_wrt_candidates=ignore_wrt_candidates,
                 iou_calculator=iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :5]
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            assert NotImplementedError
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

@BOXES.register_module()
class ATSSAssignerRbbox():
    def __init__(self,
                 topk,
                 iou_calculator=dict(type='RBboxOverlaps2D'),
                 assigned_labels_filled=0,
                 ):
        self.topk = topk
        self.iou_calculator = build_from_cfg(iou_calculator, BOXES)
        self.assigned_labels_filled = assigned_labels_filled
    
    def assign(self, bboxes, num_level_bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """        
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')

        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = jt.full((num_bboxes, ), 0).int()
        
        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]
        offsets = bboxes_points[:, None, :] - gt_points[None, :, :]
        distances = offsets.sqr().sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = jt.concat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, jt.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        def __std(x: jt.Var, dim=None):
            if dim is None:
                return jt.std(x)
            out = (x - jt.mean(x, dim=dim, keepdims=True)).sqr().sum(dim=dim, keepdims=False)
            out = out / (x.shape[dim] - 1)
            out = out.maximum(1e-6).sqrt()
            return out
        overlaps_std_per_gt = __std(candidate_overlaps, dim=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        inside_flag = points_in_rotated_boxes(bboxes_points, gt_bboxes)
        is_in_gts = inside_flag[candidate_idxs, jt.arange(num_gt)]

        is_pos = is_pos & is_in_gts

        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        INF = 100000000
        overlaps_inf = jt.transpose(jt.full_like(overlaps, -INF)).view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = jt.transpose(overlaps).view(-1)[index]
        overlaps_inf = jt.transpose(overlaps_inf.view(num_gt, -1))

        argmax_overlaps, max_overlaps = jt.argmax(overlaps_inf, dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = jt.full((num_bboxes, ), self.assigned_labels_filled, dtype=assigned_gt_inds.dtype)
            pos_inds = jt.nonzero(assigned_gt_inds > 0).squeeze(-1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

@BOXES.register_module()
class ConvexAssigner:
    """Assign a corresponding gt bbox or background to each bbox. Each
    proposals will be assigned with `0` or a positive integer indicating the
    ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        scale (float): IoU threshold for positive bboxes.
        pos_num (float): find the nearest pos_num points to gt center in this
        level.
    """

    def __init__(self, scale=4, pos_num=3, assigned_labels_filled=0):
        self.scale = scale
        self.pos_num = pos_num
        self.assigned_labels_filled = assigned_labels_filled

    def get_horizontal_bboxes(self, gt_rbboxes):
        """get_horizontal_bboxes from polygons.

        Args:
            gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).

        Returns:
            gt_rect_bboxes (torch.Tensor): The horizontal bboxes, shape (k, 4).
        """
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin = gt_xs.min(1)
        gt_ymin = gt_ys.min(1)
        gt_xmax = gt_xs.max(1)
        gt_ymax = gt_ys.max(1)
        gt_rect_bboxes = jt.concat([
            gt_xmin[:, None], gt_ymin[:, None],
            gt_xmax[:, None], gt_ymax[:, None]], dim=1)

        return gt_rect_bboxes

    def assign(self,
               points,
               gt_rbboxes,
               gt_rbboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            points (torch.Tensor): Points to be assigned, shape(n, 18).
            gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
            gt_rbboxes_ignore (Tensor, optional): Ground truth polygons that
                are labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_points = points.shape[0]
        num_gts = gt_rbboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=jt.int32)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=jt.int32)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = jt.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        assert gt_rbboxes.size(1) == 8, 'gt_rbboxes should be (N * 8)'
        gt_bboxes = self.get_horizontal_bboxes(gt_rbboxes)

        # assign gt rbox
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min_v=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((jt.log2(gt_bboxes_wh[:, 0] / scale) +
                          jt.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = jt.clamp(gt_bboxes_lvl, min_v=lvl_min, max_v=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = jt.zeros((num_points, ), dtype=jt.int32)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = jt.full((num_points, ), float('inf'))
        points_range = jt.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and
            #   all points in this level
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = jt.topk(
                points_gt_dist, self.pos_num, largest=False)
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]

            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[
                less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[
                less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = jt.full((num_points, ), self.assigned_labels_filled, dtype=jt.int32)
            pos_inds = jt.nonzero(assigned_gt_inds > 0).squeeze(-1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

@BOXES.register_module()
class MaxConvexIoUAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox. Each
    proposals will be assigned with `-1`, or a semi-positive integer indicating
    the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            points (torch.Tensor): Points to be assigned, shape(n, 18).
            gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
            overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_rbboxes_ignore (Tensor, optional): Ground truth polygons that
                are labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            assert NotImplementedError
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result
