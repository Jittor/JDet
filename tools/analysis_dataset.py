import numpy as np 
import pickle 
import os
import glob
from shapely.geometry import Polygon
from tqdm import tqdm


def read_fair(ann_file):
    data = pickle.load(open(ann_file,"rb"))
    classnames = data["cls"]
    anns = data["content"]
    breakpoint()
    print(type(data))
    print(len(data))

def iou_poly(poly1,poly2):
    poly1 = Polygon(poly1.reshape(4,2))
    poly2 = Polygon(poly2.reshape(4,2))
    if poly1.area<16 or poly2.area<16:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    iou = inter_area/max(poly1.area+poly2.area-inter_area,0.01)
    return iou

def poly_iou_np(polys1,polys2):
    ious = []
    for p1 in polys1:
        iou1 = []
        for p2 in polys2:
            iou = iou_poly(p1,p2)
            iou1.append(iou)
        ious.append(iou1)
    return np.array(ious)

def read_dota(label_path):
    files = glob.glob(os.path.join(label_path,"*.txt"))
    all_ious = []
    for i,f in tqdm(enumerate(files),total=len(files)):
        with open(f) as ff:
            lines = [line.strip().split(" ") for line in ff.readlines()]
            polys = []
            classnames = []
            for line in lines:
                if len(line)<10:
                    continue
                poly = list(map(float,line[:8]))
                classname = line[8]
                diff = int(line[9])

                polys.append(poly)
                classnames.append(classname)
            
            polys = np.array(polys)
            cc = list(set(classnames))
            labels = np.array([cc.index(c) for c in classnames])
            for l in np.unique(labels):
                ps = polys[labels==l]
                ious = poly_iou_np(ps,ps)
                ious = ious[ious<0.95].reshape(-1)
                all_ious.append(ious)

            # if i>1000:
            #     break

    all_ious = np.concatenate(all_ious)
    print(all_ious.max())
    all_ious = np.sort(all_ious)
    print(all_ious[-200:])



    


def main():
    ann_file = "/data/lxl/dataset/split_ms_fair/trainval/annfiles/ori_annfile.pkl"
    dota_label_dir = "/data/lxl/dataset/fair_DOTA/trainval/labelTxt"
    # read_fair(ann_file)
    read_dota(dota_label_dir)
    

if __name__ == "__main__":
    main()