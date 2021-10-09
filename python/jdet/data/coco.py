from jdet.utils.general import build_file
from jittor.dataset import Dataset 
import warnings 
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    warnings.warn("pycocotools is not installed!")
import os 
from PIL import Image
import numpy as np 
import json 
import warnings
import itertools
from collections import OrderedDict
from terminaltables import AsciiTable


from jdet.utils.registry import DATASETS
from jdet.config import COCO_CLASSES
from jdet.data.transforms import Compose

@DATASETS.register_module()
class COCODataset(Dataset):
    """ COCO Dataset.
    Args:
        root(str): the image root path.
        anno_file(str): the annotation file.
        transforms(list): the transforms for dataset, it can be list(dict) or list(transform), default None.
        batch_size(int): default 1.
        num_workers(int): default 0.
        shuffle(bool): default False.
        drop_last(bool): drop the last batch if len(batch) % gpu_nums !=0, must be True when use multi gpus, default False.
        filter_empty_gt(bool): filter the image without groundtruth, default True.
        use_anno_cats(bool): use the classnames from annotation file instead of COCO_CLASSES(80), default False.
    """

    CLASSES = COCO_CLASSES
    
    def __init__(self,root,anno_file,transforms=None,batch_size=1,num_workers=0,shuffle=False,drop_last=False,filter_empty_gt=True,use_anno_cats=False):
        super(COCODataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,drop_last=drop_last)
        self.root = root 
        self.coco = COCO(anno_file)
        
        if isinstance(transforms,list):
            transforms = Compose(transforms)
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be list or callable")

        self.transforms = transforms
        
        if use_anno_cats:
            self.CLASSES = [cat['name'] for cat in self.coco.cats.values()]

        self.cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = list(sorted(self.coco.imgs.keys()))

        if filter_empty_gt:
            self.img_ids = self._filter_imgs()

        self.total_len = len(self.img_ids)

    def _filter_imgs(self):
        """Filter images without ground truths."""
        # reference mmdetection
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        ids_in_cat &= ids_with_ann

        tmp_img_ids = [img_id for img_id in self.img_ids if img_id in ids_in_cat]

        img_ids = []
        for img_id in tmp_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)

            # remove ignored or crowd box
            anno = [obj for obj in anno if ("is_crowd" not in obj or obj["iscrowd"] == 0) and ("ignore" not in obj or obj["ignore"] == 0) ]
            # if it's empty, there is no annotation
            if len(anno) == 0:
                continue
            # if all boxes have close to zero area, there is no annotation
            if all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno):
                continue
            img_ids.append(img_id)

        # sort indices for reproducible results
        img_ids = sorted(img_ids)

        return img_ids

    def _read_ann_info(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        width,height = image.size 
        assert width == img_info['width'] and height == img_info["height"],"image size is different from annotations"

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

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

        ann = dict(
            img_id = img_id,
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            classes=self.CLASSES,
            ori_img_size=(width,height),
            img_size=(width,height))

        return image,ann
    

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image,anno = self._read_ann_info(img_id)

        if self.transforms is not None:
            image, anno = self.transforms(image, anno)

        return image, anno 

    def collate_batch(self,batch):
        imgs = []
        anns = []
        max_width = 0
        max_height = 0
        for image,ann in batch:
            height,width = image.shape[-2],image.shape[-1]
            max_width = max(max_width,width)
            max_height = max(max_height,height)
            imgs.append(image)
            anns.append(ann)
        N = len(imgs)
        batch_imgs = np.zeros((N,3,max_height,max_width),dtype=np.float32)
        for i,image in enumerate(imgs):
            batch_imgs[i,:,:image.shape[-2],:image.shape[-1]] = image
        
        return batch_imgs,anns 

    
    def save_results(self,results,save_file):
        """Convert detection results to COCO json style."""
        def xyxy2xywh(box):
            x1,y1,x2,y2 = box.tolist()
            return [x1,y1,x2-x1,y2-y1]
        
        json_results = []
        for result,target in results:
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
            predictions = json.load(open(save_file))
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


def test_cocodataset():
    dataset = COCODataset(root="/mnt/disk/lxl/dataset/coco/images/val2017",anno_file="/mnt/disk/lxl/dataset/coco/annotations/instances_val2017.json")
    print(len(dataset.CLASSES))
    print(len(dataset.cat_ids))


if __name__ == "__main__":
    test_cocodataset()
