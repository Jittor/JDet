import os
import xml.etree.cElementTree as ET
from tqdm import tqdm
import cv2
import numpy as np

def xml2txt(xml_path, txt_path, rescale, plus):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    object=root.findall("object")
    out_lines = []
    for ob in object:
        if plus:
            box=ob.find("rotated_bndbox")
            x1 = str(float(box.find("x1").text) * rescale[0])
            y1 = str(float(box.find("y1").text) * rescale[1])
            x2 = str(float(box.find("x2").text) * rescale[0])
            y2 = str(float(box.find("y2").text) * rescale[1])
            x3 = str(float(box.find("x3").text) * rescale[0])
            y3 = str(float(box.find("y3").text) * rescale[1])
            x4 = str(float(box.find("x4").text) * rescale[0])
            y4 = str(float(box.find("y4").text) * rescale[1])
        else:
            box=ob.find("bndbox")
            xmin = str(float(box.find("xmin").text) * rescale[0])
            ymin = str(float(box.find("ymin").text) * rescale[1])
            xmax = str(float(box.find("xmax").text) * rescale[0])
            ymax = str(float(box.find("ymax").text) * rescale[1])
            x1 = xmin
            y1 = ymin
            x2 = xmin
            y2 = ymax
            x3 = xmax
            y3 = ymax
            x4 = xmax
            y4 = ymin
        name = str(ob.find('name').text)
        diff = ob.find("difficult").text
        data = x1 + " " + y1 + " "+x2 + " " + y2 + " "+x3 + " " + y3 + " "+x4 + " " + y4 + " "+name+' '+diff+"\n"
        out_lines.append(data)

    f = open(txt_path, "w")
    f.writelines(out_lines)
    f.close()

def ssdd_to_dota(img_path, anno_path, target_path, resize, plus):
    names = []
    for root, dirs, files in os.walk(img_path):
        for name in files:
            if not name.endswith(".jpg"):
                continue
            names.append(name[:-4])
    out_img_path = os.path.join(target_path, "images")
    out_anno_path = os.path.join(target_path, "labelTxt")
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_anno_path, exist_ok=True)
    for i in tqdm(range(len(names))):
        name = names[i]
        img = cv2.imread(os.path.join(img_path, name+".jpg"))
        h, w, _ = img.shape
        img = cv2.resize(img, (resize, resize))
        cv2.imwrite(os.path.join(out_img_path, name+".png"), img)
        xml2txt(os.path.join(anno_path, name+'.xml'), os.path.join(out_anno_path, name+'.txt'), (resize / w, resize / h), plus)