import jittor as jt 
from jdet.config.constant import DOTA1_CLASSES
from jdet.utils.general import check_dir
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.data.devkits.result_merge import mergebypoly
import os

def prepare(result_pkl,save_path):
    check_dir(save_path)
    results = jt.load(result_pkl)
    data = {}
    for result,target in results:
        dets,labels = result
        print(dets.shape)
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

def test():
    result_pkl = "/home/lxl/workspace/JDet/work_dirs/s2anet_r50_fpn_1x_dota/test/test_12.pkl"
    save_path = "/home/lxl/workspace/JDet/work_dirs/s2anet_r50_fpn_1x_dota/submit/before_nms"
    final_path = "/home/lxl/workspace/JDet/work_dirs/s2anet_r50_fpn_1x_dota/submit/after_nms"
    prepare(result_pkl,save_path)
    check_dir(final_path)
    mergebypoly(save_path,final_path)

if __name__ == "__main__":
    test()
