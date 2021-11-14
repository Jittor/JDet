import os
import glob
from jdet.data.devkits.result_merge import py_cpu_nms_poly_fast
from jdet.data.devkits.dota_to_fair import dota_to_fair
from multiprocessing import Pool
from functools import partial
import zipfile
import shutil
import numpy as np

def merge_file(src_file,dst_path,nms_op,nms_thr=0.1):
    name = os.path.split(src_file)[-1]
    os.makedirs(dst_path,exist_ok=True)
    dst_file = os.path.join(dst_path,name)
    with open(src_file,"r") as f:
        result = {}
        lines = [x.strip().split(' ') for x in f.readlines()]
        for splitline in lines:     
            oriname = splitline[0]
            det = list(map(float, splitline[1:]))
            det = det[1:]+det[:1]
            if oriname not in result:
                result[oriname] = []
            result[oriname].append(det)
        outs = []
        for img_id,dets in result.items():
            dets = np.array(dets)
            keep = nms_op(dets,nms_thr)
            dets = dets[keep].tolist()
            for det in dets:
                confidence = det[-1]
                bbox = det[0:-1]
                outline = img_id + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox)) +"\n"
                outs.append(outline)
        with open(dst_file,"w") as fo:
            fo.writelines(outs)        

def merge_files(src_path,dst_path,nms_op = py_cpu_nms_poly_fast, nms_thr=0.1, process_num=37):
    files = glob.glob(os.path.join(src_path,"*.txt"))
    merge_fn = partial(merge_file,dst_path=dst_path,nms_op=nms_op,nms_thr=nms_thr)
    if process_num<2:
        list(map(merge_fn,files))
    else:
        pool = Pool(process_num)
        pool.map(merge_fn,files)

def merge_src_files(src_paths,dst_path):
    os.makedirs(dst_path,exist_ok=True)
    for ipath in src_paths:
        files = glob.glob(ipath+"/*.txt")
        for ff in files:
            filename = ff.split("/")[-1]
            filename = filename.replace("Task1_","")
            wf = open(os.path.join(dst_path,filename),"a")
            with open(ff) as f:
                for line in f.readlines():
                    wf.write(line)
            wf.close()
    
def build_path(save_dir,ext):
    path = os.path.join(save_dir,ext)
    if os.path.exists(path):
        shutil.rmtree(path)
    return  path 

def merge_results(result_dirs,save_dir,name="tmp", image_dirs=None, dataset_type="dota"):
    before_nms_path = build_path(save_dir,"before_nms")
    after_nms_path = build_path(save_dir,"after_nms")
    zip_path = os.path.join(save_dir, name + ".zip")
    
    print("Merge files...")
    merge_src_files(result_dirs,before_nms_path)
    print("Merge results...")
    merge_files(before_nms_path,after_nms_path)

    if dataset_type.lower() == "fair":
        print("converting to fair...")
        assert image_dirs is not None
        fair_path = build_path(save_dir,"final_fair")
        dota_to_fair(after_nms_path,fair_path,image_dirs)
        after_nms_path = fair_path
    else:
        assert dataset_type.lower() == "dota"
    
    print("zip..")
    files = glob.glob(os.path.join(after_nms_path,"*"))
    with zipfile.ZipFile(zip_path, 'w',zipfile.ZIP_DEFLATED) as t:
        for f in files:
            if dataset_type == "fair":
                t.write(f, os.path.join("test",os.path.split(f)[-1]))
            else:
                t.write(f, "Task1_"+os.path.split(f)[-1])

def main():
    result_dirs = [
        "temp/orpn_fair",
        "temp/merge7_fair",
    ]
    name = "fair_merge16"
    image_dirs = "/mnt/disk/lxl/dataset/fair_1024/test_1024_200_0.5-1.0-1.5/images/"
    dataset_type = "fair"

    save_dir = "temp"
    merge_results(result_dirs,save_dir,name=name,image_dirs = image_dirs, dataset_type=dataset_type)


if __name__ == "__main__":
    main()