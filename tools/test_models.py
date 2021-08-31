import os
tasks = ["faster_rcnn", "retinanet", "s2anet", "ssd"]

if (not os.path.exists("test_datas.zip")):
    os.system("wget https://cloud.tsinghua.edu.cn/f/c790a39cfdd7453c9732/?dl=1")
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
    cmd = f"cd projects/{task} && python test_{task}.py"
    os.system(cmd)
