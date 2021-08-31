import os

tasks = ["faster_rcnn", "retinanet","s2anet","ssd"]
if (not os.path.exists("test_datas.zip")):
    os.system("wget https://cloud.tsinghua.edu.cn/f/63b1224dbfc6464495d5/?dl=1")
for task in tasks:
    print(f"===============testing {task}==================")
    cmd = f"cd projects/{task} && python test_{task}.py"
    os.system(cmd)
