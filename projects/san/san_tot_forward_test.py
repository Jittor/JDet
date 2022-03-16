import jittor as jt
import copy

jt.set_global_seed(0)
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
import numpy as np
import random
import jdet
import argparse
import os
import pickle as pk
from jdet.runner import Runner 
from PIL import Image
from jittor.transform import Resize, CenterCrop, ImageNormalize, to_tensor

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(0)
    np.random.seed(0)
    random.seed(0)
    init_cfg("/home/flowey/remote2/JDet/configs/san10_pairwise.py")
    runner = Runner()
    
    # image_file = '/home/flowey/dataset/ILSVRC2012/img_test/n01440764/ILSVRC2012_val_00009379.JPEG'
    # image_torch_file = '/home/flowey/remote2/JDet/projects/san/test_datas_san/single_image.pkl'
    # val_trans = runner.val_dataset.transforms.transforms
    # image = Image.open(image_file).convert('RGB')
    # images = []
    # images.append(copy.deepcopy(np.array(image, dtype=np.float64)))
    # for t in val_trans:
    #     image, _ = t(image, dict())
    #     if isinstance(image, jt.Var):
    #         images.append(copy.deepcopy(image.numpy()))
    #     elif isinstance(image, np.ndarray):
    #         images.append(copy.deepcopy(image))
    #     else:
    #         images.append(copy.deepcopy(np.array(image, dtype=np.float64)))
    # images.append(copy.deepcopy(np.array(images[2]).transpose((2,0,1))))
    # t = images[3]
    # images[3] = images[4]
    # images[4] = t
    # images_torch = pk.load(open(image_torch_file, 'rb'))
    # p = np.argmax(np.abs((images_torch[4] - images[4])/(images[4]+1e-10)))
    # i, j, k = p //(images[4].shape[1]*images[4].shape[2]), (p // images[4].shape[2]) % images[4].shape[1], p % images[4].shape[2]
    # print(images_torch[3][i,j,k] * 255, images[3][i,j,k])
    # print(images_torch[4][i,j,k], images[4][i,j,k])
    # print(i,j,k)
    # print(np.max(np.abs((images_torch[4] - images[4])/(images[4]+1e-10))))

    numpy_save_dir = '/home/flowey/remote2/JDet/projects/san/test_datas_san/models_numpy.pth'
    numpy_dict = pk.load(open(numpy_save_dir, 'rb'))
    jittor_dict = dict()
    for k, v in numpy_dict.items():
        jittor_dict[k] = jt.array(v)
    runner.model.load_state_dict(jittor_dict)
    runner.val()

    
if __name__ == "__main__":
    main()