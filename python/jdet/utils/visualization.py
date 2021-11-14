import os 
import glob 
import numpy as np
from jdet.utils.draw import draw_bboxes
from jdet.config.constant import DOTA1_CLASSES, DOTA_COLORS
from tqdm import tqdm

def read_dota(dota_dir):
    files = glob.glob(os.path.join(dota_dir,"*.txt"))
    results = {}
    class_names = []
    for i,f in enumerate(files):
        classname = os.path.split(f)[-1].split(".txt")[0]
        classname  = classname.replace("Task1_","")
        class_names.append(classname)
        with open(f) as ff:
            for line in ff.readlines():
                line = line.strip().split(" ")
                img_id = line[0]
                s_poly = [i]+[float(p) for p in line[1:]]
                if img_id not in results:
                    results[img_id] = []
                results[img_id].append(s_poly)
    dets = {}
    for k,d in results.items():
        d = np.array(d,dtype=np.float32)
        labels,scores,polys = d[:,0],d[:,1],d[:,2:]
        labels = labels.astype(np.int32)
        dets[k] = (polys,scores,labels)
    
    return dets,class_names


def visualize_dota(dota_dir,image_dir,save_dir):
    dets,class_names = read_dota(dota_dir)
    os.makedirs(save_dir,exist_ok=True)
    for img_id,(polys,scores,labels) in dets.items():
        img_file = os.path.join(image_dir,img_id+".png")
        if not os.path.exists(img_file):
            print(img_file,"not exists.")
            continue
        save_file = os.path.join(save_dir,img_id+".png")
        draw_bboxes(img_file,
                  polys,
                  labels=labels,
                  scores=scores,
                  class_names=class_names,
                  score_thr=0.5,
                  colors='green',
                  thickness=1,
                  with_text=True,
                  font_size=10,
                  out_file=save_file)

def visualize_results(results,classnames,files,save_dir,**kwargs):
    os.makedirs(save_dir,exist_ok=True)
    for (bboxes, scores, labels),img_file in tqdm(zip(results,files)):
        save_file = os.path.join(save_dir,os.path.split(img_file)[-1])
        draw_bboxes(img_file,bboxes,labels=labels,scores=scores,class_names=classnames,out_file=save_file,**kwargs)

def visualize_dota_ground_truth(gt_dir, classnames, save_dir,style=0):
    img_dir = os.path.join(gt_dir, "images")
    anno_dir = os.path.join(gt_dir, "labelTxt")
    assert(os.path.exists(img_dir))
    assert(os.path.exists(anno_dir))
    assert(style in [1,2])
    label_dict = {}
    for i in range(len(classnames)):
        label_dict[classnames[i]] = i
    
    names = []
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if not f.endswith(".png"):
                continue
            names.append(f[:-4])
    results = []
    files = []
    for i in tqdm(range(len(names))):
        files.append(os.path.join(img_dir, names[i] + '.png'))
        datas = open(os.path.join(anno_dir, names[i] + '.txt')).readlines()
        bboxes = []
        scores = []
        labels = []
        for data in datas:
            ds = data.split(" ")
            if len(ds) < 10:
                continue
            bboxes.append([int(i) for i in ds[:8]])
            scores.append(1)
            labels.append(label_dict[ds[8]])
        if len(bboxes) == 0:
            bboxes = np.zeros([0,8], dtype=np.float32)
            scores = np.zeros([0], dtype=np.float32)
            labels = np.zeros([0], dtype=np.int32)
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
        results.append((bboxes, scores, labels))
    if style == 1:
        visualize_results(results, classnames, files, save_dir,thickness=2)
    elif style == 2:
        visualize_results(results, classnames, files, save_dir, colors=DOTA_COLORS, with_text=False,thickness=2)

def main():
    gt_dir = "/home/cxjyxx_me/workspace/JAD/datasets/DOTA/train"
    save_dir = "./temp"
    visualize_dota_ground_truth(gt_dir, DOTA1_CLASSES, save_dir, style=1)

    # dota_dir = "/home/cxjyxx_me/workspace/JAD/SAR/JDet/projects/gliding/submit_zips/temp2"
    # image_dir = "/home/cxjyxx_me/workspace/JAD/datasets/DOTA/test/images"
    # save_dir = "./temp"
    # visualize_dota(dota_dir,image_dir,save_dir)

if __name__ == "__main__":
    main()