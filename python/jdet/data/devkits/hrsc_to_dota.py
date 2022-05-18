import os
import os.path as osp
import xml.etree.cElementTree as ET
from tqdm import tqdm
from jdet.data.devkits.conver_hrsc_to_mmdet import list_from_file
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
import cv2

def xml2txt(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    out_lines = []
    for obj in root.findall('HRSC_Objects')[0].findall('HRSC_Object'):
        label = 'ship'
        bbox = []
        for key in ['mbox_cx', 'mbox_cy', 'mbox_w', 'mbox_h', 'mbox_ang']:
            bbox.append(obj.find(key).text)
        poly = rotated_box_to_poly_single(bbox)
        difficult = int(obj.find('difficult').text)
        temp_txt = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7],
            label, difficult
        )
        out_lines.append(temp_txt)

    f = open(txt_file, "w")
    f.writelines(out_lines)
    f.close()

def hrsc_to_dota(img_path, xml_path, ann_file, out_path, convert_label=True, ext='.bmp'):
    out_img_path = osp.join(out_path, "images")
    out_anno_path = osp.join(out_path, "labelTxt")
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_anno_path, exist_ok=True)
    img_ids = list_from_file(ann_file)
    for img_id in tqdm(img_ids):
        # TODO: add process, or replace with copy
        img = cv2.imread(osp.join(img_path, f'{img_id}{ext}'))
        cv2.imwrite(osp.join(out_img_path, f'{img_id}.png'), img)
        if (convert_label):
            xml_file = osp.join(xml_path, f'{img_id}.xml')
            txt_file = osp.join(out_anno_path, f'{img_id}.txt')
            xml2txt(xml_file, txt_file)