
import cv2
import os
import shutil
from PIL import Image
import pywebio as pw
import jittor as jt
jt.flags.use_cuda_managed_allocator=1
jt.flags.use_cuda=1
from jdet.runner import Runner 
from jdet.config import init_cfg, get_cfg, update_cfg

def process():
    tmp_path = "tmp_demo"
    config_path = "configs/s2anet/s2anet_r50_fpn_1x_dota.py"
    weights_path = "weights/s2anet_r50_fpn_1x_dota.pkl"
    # init
    pw.output.clear("out")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    in_path = os.path.join(tmp_path, "images_in")
    out_path = os.path.join(tmp_path, "images_out")
    os.makedirs(in_path)
    os.makedirs(out_path)
    # copy image
    img_path = pw.pin.pin.img_path
    if not os.path.exists(img_path):
        print("img not exist:", img_path)
        return
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(in_path, img_name))
    img = cv2.imread(os.path.join(in_path, img_name))
    img = cv2.resize(img, (256,256))
    with pw.output.use_scope('out'):
        pw.output.put_text("输入图像（缩略图）：")
        pw.output.put_image(Image.fromarray(img))
    # run
    init_cfg(config_path)
    cfg = get_cfg()
    dataset_cfg = cfg.dataset
    dataset_cfg["test"]["images_dir"] = in_path
    update_cfg({"resume_path":weights_path, "dataset":dataset_cfg})
    runner = Runner()
    runner.run_on_images(out_path, score_thr=0.5)
    #out
    img_out = cv2.imread(os.path.join(out_path, img_name))
    with pw.output.use_scope('out'):
        pw.output.put_text("输出图像：")
        pw.output.put_image(Image.fromarray(img_out))
    print("done")

def web_server():
    pw.output.put_text("示例图像：")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2795__1.0__0___0.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2785__1.0__3296___0.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2763__1.0__824___824.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2795__1.0__1648___1648.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2795__1.0__3296___824.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2757__1.0__0___0.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2757__1.0__824___1452.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2763__1.0__1648___3563.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2763__1.0__2472___3296.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2785__1.0__1648___1648.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2798__1.0__1398___1356.png")
    pw.output.put_text("/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/P2803__1.0__601___0.png")
    pw.output.put_text("============")
    pw.pin.put_input("img_path", label="图像路径")
    pw.output.put_buttons(['Gen image'], lambda _: process())

pw.start_server(web_server, port=8123)