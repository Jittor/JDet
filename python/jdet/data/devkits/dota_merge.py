import shutil
import jittor as jt 
from jdet.config.constant import DOTA1_CLASSES
from jdet.utils.general import check_dir
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.data.devkits.result_merge import mergebypoly
import os
import shutil
from tqdm import tqdm
import numpy as np

def prepare(result_pkl,save_path):
    check_dir(save_path)
    results = jt.load(result_pkl)
    data = {}
    for result,target in tqdm(results):
        dets,labels = result
        img_name = os.path.splitext(os.path.split(target["img_file"])[-1])[0]
        for det,label in zip(dets,labels):
            bbox = det[:5]
            score = det[5]
            classname = DOTA1_CLASSES[label]
            bbox = rotated_box_to_poly_single(bbox)
            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                        img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                        bbox[5], bbox[6], bbox[7])
            if classname not in data:
                data[classname] = []
            data[classname].append(temp_txt)
    for classname,lines in data.items():
        f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
        f_out.writelines(lines)
        f_out.close()

def prepare_gliding(result_pkl,save_path):
    check_dir(save_path)
    results = jt.load(result_pkl)
    data = {}
    for result,target in tqdm(results):
        img_name = os.path.splitext(os.path.split(target["img_file"])[-1])[0]
        for bbox,score,label in zip(*result):
            classname = DOTA1_CLASSES[label-1]
            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                        img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                        bbox[5], bbox[6], bbox[7])
            if classname not in data:
                data[classname] = []
            data[classname].append(temp_txt)
    for classname,lines in data.items():
        f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
        f_out.writelines(lines)
        f_out.close()

def prepare_fasterrcnn(result_pkl,save_path):
    check_dir(save_path)
    results = jt.load(result_pkl)
    data = {}
    for result,target in tqdm(results):
        img_name = os.path.splitext(os.path.split(target['img_meta'][0]["img_file"])[-1])[0]
        for idx, res in enumerate(result):
            for i in range(res.shape[0]):
                bbox = res[i]
                classname = DOTA1_CLASSES[idx]
                temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            img_name, bbox[-1], bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                            bbox[5], bbox[6], bbox[7])
                if classname not in data:
                    data[classname] = []
                data[classname].append(temp_txt)
    for classname,lines in data.items():
        f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
        f_out.writelines(lines)
        f_out.close()

def dota_merge(result_pkl, save_path, final_path):
    if "gliding" in result_pkl:
        prepare_gliding(result_pkl,save_path)
    elif "faster_rcnn" in result_pkl:
        prepare_fasterrcnn(result_pkl,save_path)
    else:
        prepare(result_pkl,save_path)
    check_dir(final_path)
    mergebypoly(save_path,final_path)

def dota_merge_result(result_pkl,work_dir,epoch,name):
    print("Merge results...")
    save_path = os.path.join(work_dir, f"test/submit_{epoch}/before_nms")
    final_path = os.path.join(work_dir, f"test/submit_{epoch}/after_nms")
    zip_path = os.path.join("submit_zips", name + ".zip")
    if (os.path.exists(save_path)):
        shutil.rmtree(save_path)
    if (os.path.exists(final_path)):
        shutil.rmtree(final_path)
    if (os.path.exists(zip_path)):
        os.remove(zip_path)
    if not os.path.exists("submit_zips"):
        os.makedirs("submit_zips")
    dota_merge(result_pkl, save_path, final_path)
    os.system(f"zip -rj {zip_path} {os.path.join(final_path,'*')}")


if __name__ == "__main__":
    work_dir = "/mnt/disk/lxl/JDet/work_dirs/gliding_r50_fpn_1x_dota_bs2_tobgr_steplr_norotate_ms"
    epoch = 12
    result_pkl = f"{work_dir}/test/test_{epoch}.pkl"
    save_path = f"{work_dir}/test/submit_{epoch}/before_nms"
    final_path = f"{work_dir}/test/submit_{epoch}/after_nms"
    dota_merge(result_pkl, save_path, final_path)