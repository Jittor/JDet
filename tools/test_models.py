import os
tasks = ["roi_transformer", "faster_rcnn", "retinanet", "s2anet", "ssd", "gliding", 
"oriented_rcnn", "fcos", "yolo"]
zip_path = "https://cloud.tsinghua.edu.cn/f/a5bcad6d48bc42a38c06/?dl=1"

if (not os.path.exists("test_datas.zip")):
    os.system(f"wget {zip_path}")
    os.system("mv 'index.html?dl=1' test_datas.zip")
    os.system("unzip -o test_datas.zip")
    for task in tasks:
        src_path = os.path.join(os.path.abspath('.'), f"test_datas/test_datas_{task}")
        tar_path = os.path.join(os.path.abspath('.'), f"projects/{task}/test_datas_{task}")
        if (os.path.exists(tar_path) or os.path.islink(tar_path)):
            os.system(f"rm -rf {tar_path}")
        os.system(f"ln -s {src_path} {tar_path}")
for task in tasks:
    print(f"===============testing {task}==================")
    cmd = f"cd projects/{task} && python3 test_{task}.py"
    os.system(cmd)
