from jdet.utils.general import build_file
from jittor.dataset import Dataset 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os 
from PIL import Image
import numpy as np 
import json 
import warnings
import itertools
from collections import OrderedDict
from terminaltables import AsciiTable
import jittor as jt

from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_CLASSES
from .transforms import Compose

@DATASETS.register_module()
class DOTARCNNDataset(Dataset):
    CLASSES = DOTA1_CLASSES

    def __init__(self,root,anno_file,transforms=None,batch_size=1,num_workers=0,shuffle=False,drop_last=False,filter_empty_gt=True,use_anno_cats=False,test_mode=False,keep_flip=True):
        super().__init__()
        print('init DOTARCNNDataset')
        self.root = root 
        self.coco = COCO(anno_file)
        self.keep_flip = keep_flip
        self.test_mode = test_mode
        
        if isinstance(transforms,list):
            transforms = Compose(transforms)
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be list or callable")

        self.transforms = transforms
        
        if use_anno_cats:
            self.CLASSES = [cat['name'] for cat in self.coco.cats.values()]

        self.cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        self.img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            self.img_infos.append(info)

        if not self.test_mode:
            if filter_empty_gt:
                valid_inds = self._filter_imgs()
                self.img_infos = [self.img_infos[i] for i in valid_inds]
                self.img_ids = [self.img_ids[i] for i in valid_inds]

        self.total_len = len(self.img_ids)

        if not self.test_mode:
            self._set_group_flag()
        
        self.set_attrs(batch_size = batch_size, total_len = self.total_len, shuffle = shuffle, drop_last = drop_last, keep_numpy_array=1)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _read_ann_info(self, img_id, with_mask=True):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        width,height = image.size 
        assert width == img_info['width'] and height == img_info["height"],"image size is different from annotations"

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)

        if len(gt_bboxes) == 0:
            return None

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        
        flip = True if np.random.rand() < 0.5 else False
        
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=(width,height,3),
            pad_shape=(width,height,3),
            scale_factor=1.0,
            flip=flip,
            img_file=img_info["file_name"])

        if self.transforms is not None:
            image, img_meta = self.transforms(image, img_meta)

        if self.keep_flip and flip:
            image = np.flip(image, -1)
            gt_bboxes_ = gt_bboxes.copy()
            gt_bboxes_[:,0] = 1023 - gt_bboxes[:,2]
            gt_bboxes_[:,2] = 1023 - gt_bboxes[:,0]
            gt_bboxes = gt_bboxes_
            gt_masks = [mask[:, ::-1] for mask in gt_masks]

        data = dict(
            img=image,
            img_meta=img_meta,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks
        )

        # from numpy import asarray   
        # print(asarray(image))
        # print(gt_bboxes, gt_labels)
        return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(self.total_len, dtype=np.uint8)
        for i in range(self.total_len):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_id = self.img_ids[idx]
        return self._read_ann_info(img_id)

    def prepare_test_img(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ori_shape = (img_info['height'], img_info['width'], 3)

        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=ori_shape,
            pad_shape=ori_shape,
            scale_factor=1.0,
            flip=False,
            img_file=img_info["file_name"])

        if self.transforms is not None:
            image, img_meta = self.transforms(image, img_meta)

        data = dict(
            img=image,
            img_meta=img_meta
        )
        return data

    def collate_batch(self, batch):
        img = []
        img_meta = []
        if not self.test_mode:
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_masks = []
        for data in batch:
            img.append(data['img'])
            img_meta.append(data['img_meta'])
            if not self.test_mode:
                gt_bboxes.append(jt.array(data['gt_bboxes']))
                gt_labels.append(jt.array(data['gt_labels']))
                gt_bboxes_ignore.append(jt.array(data['gt_bboxes_ignore']))
                gt_masks.append(np.stack(data['gt_masks'], axis=0))
        img = jt.array(np.stack(img, axis=0))
        if not self.test_mode:
            targets = dict(
                img_meta=img_meta, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks
            )
        else:
            targets = [dict(
                img_meta=img_meta
            )]
        return img, targets
    
    def save_results(self,results,save_file):
        """Convert detection results to COCO json style."""
        def xyxy2xywh(box):
            x1,y1,x2,y2 = box.tolist()
            return [x1,y1,x2-x1,y2-y1]
        
        json_results = []
        for result in results:
            img_id = result["img_id"]
            for box,score,label in zip(result["boxes"],result["scores"],result["labels"]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(box)
                data['score'] = float(score)
                data['category_id'] = self.cat_ids[int(label)-1]
                json_results.append(data)
        json.dump(json_results,open(save_file,"w"))

    
    def evaluate(self,
                 results,
                 work_dir,
                 epoch,
                 metric='bbox',
                 logger=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        save_file = build_file(work_dir,prefix=f"detections/val_{epoch}.json")
        self.save_results(results,save_file)
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
                logger.print_log(msg)

            iou_type = metric
            predictions = json.load(open(results_file))
            if iou_type == 'segm':
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                for x in predictions:
                    x.pop('bbox')
                warnings.simplefilter('once')
                warnings.warn(
                    'The key "bbox" is deleted for more accurate mask AP '
                    'of small/medium/large instances since v2.12.0. This '
                    'does not change the overall mAP calculation.',
                    UserWarning)
            if len(predictions)==0:
                warnings.warn('The testing results of the whole dataset is empty.')
                break
            cocoDt = cocoGt.loadRes(predictions)
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    if logger:
                        logger.print_log('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        return eval_results