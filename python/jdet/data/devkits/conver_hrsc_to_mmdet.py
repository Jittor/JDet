import os
import os.path as osp
import xml.etree.ElementTree as ET

import pickle
import numpy as np
from PIL import Image
from jdet.config.constant import get_classes_by_name
from tqdm import tqdm

def list_from_file(filename, prefix='', offset=0, max_num=0):
    """Load a text file and parse the content as a list of strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the begining of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list

def convert_hrsc_to_mmdet(img_path, xml_path, ann_file, out_path, convert_labels=True, filter_empty_gt=True, ext='.bmp', type="HRSC2016"):
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        image_path: path of all images
        xml_path: path for annotations in xml format
        ann_file: imageset file
        out_path: output pkl file path
        trainval: trainval or test
    """
    label_ids = {name: i + 1 for i, name in enumerate(get_classes_by_name(type))}
    img_ids = list_from_file(ann_file)
    data_dict = []
    for img_id in tqdm(img_ids):
        img = Image.open(osp.join(img_path, f'{img_id}{ext}'))
        img_info = {}
        img_info['filename'] = f'{img_id}{ext}'
        img_info['height'] = img.height
        img_info['width'] = img.width
        if convert_labels:
            xml_file = osp.join(xml_path, f'{img_id}.xml')
            if not osp.exists(xml_file):
                print(f'Annotation: {xml_file} Not Exist')
                continue
            tree = ET.parse(xml_file)
            root = tree.getroot()
            bboxes, bboxes_ignore, labels, labels_ignore = [], [], [], []
            for obj in root.findall('HRSC_Objects')[0].findall('HRSC_Object'):
                label = label_ids['ship']
                bbox = []
                for key in ['mbox_cx', 'mbox_cy', 'mbox_w', 'mbox_h', 'mbox_ang']:
                    bbox.append(obj.find(key).text)
                difficult = int(obj.find('difficult').text)
                if difficult:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            if filter_empty_gt and (len(labels)+len(labels_ignore) == 0):
                continue
            ann = {}
            ann['bboxes'] = np.array(bboxes, dtype=np.float32)
            ann['labels'] = np.array(labels, dtype=np.int64)
            ann['bboxes_ignore'] = np.array(bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(labels_ignore, dtype=np.int64)
            img_info['ann'] = ann
        data_dict.append(img_info)
    print("left images:", len(data_dict))
    pickle.dump(data_dict, open(out_path, "wb"))

