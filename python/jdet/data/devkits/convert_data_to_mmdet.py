import os
import os.path as osp

import pickle
import numpy as np
from PIL import Image

from jdet.models.boxes.box_ops import poly_to_rotated_box_single
from tqdm import tqdm
from jdet.config.constant import get_classes_by_name


def parse_ann_info(label_base_path, img_name, label_ids):
    lab_path = osp.join(label_base_path, img_name + '.txt')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            bbox = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = tuple(poly_to_rotated_box_single(bbox).tolist())
            class_name = ann_line[8]
            difficult = int(ann_line[9])
            # ignore difficult =2
            if difficult == 0:
                bboxes.append(bbox)
                labels.append(label_ids[class_name])
            elif difficult == 1:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label_ids[class_name])
    return bboxes, labels, bboxes_ignore, labels_ignore


def convert_data_to_mmdet(src_path, out_path, trainval=True, filter_empty_gt=True, ext='.png', type=''):
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output pkl file path
        trainval: trainval or test
    """
    label_ids = {name: i + 1 for i, name in enumerate(get_classes_by_name(type))}
    img_path = os.path.join(src_path, 'images')
    label_path = os.path.join(src_path, 'labelTxt')
    img_lists = os.listdir(img_path)
    img_lists.sort()

    data_dict = []
    for img in tqdm(img_lists):
        img_info = {}
        img_name = osp.splitext(img)[0]
        label = os.path.join(label_path, img_name + '.txt')
        img = Image.open(osp.join(img_path, img))
        img_info['filename'] = img_name + ext
        img_info['height'] = img.height
        img_info['width'] = img.width
        if trainval:
            if not os.path.exists(label):
                print('Label:' + img_name + '.txt' + ' Not Exist')
                continue
            # filter images without gt to speed up training
            if filter_empty_gt & (osp.getsize(label) == 0):
                continue
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(label_path, img_name, label_ids)
            ann = {}
            ann['bboxes'] = np.array(bboxes, dtype=np.float32)
            ann['labels'] = np.array(labels, dtype=np.int64)
            ann['bboxes_ignore'] = np.array(bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(labels_ignore, dtype=np.int64)
            img_info['ann'] = ann
        data_dict.append(img_info)
    print("left images:", len(data_dict))
    pickle.dump(data_dict, open(out_path, "wb"))
