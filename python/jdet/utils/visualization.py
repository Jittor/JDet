import os 
import glob 
import numpy as np
from draw import draw_bboxes

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

def main():
    dota_dir = "C:/Users/lxl/Desktop/dota_results"
    image_dir = "D:/Dataset/DOTA/test/images"
    save_dir = "C:/Users/lxl/Desktop/dota_vis"
    visualize_dota(dota_dir,image_dir,save_dir)

if __name__ == "__main__":
    main()