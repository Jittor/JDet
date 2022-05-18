import jittor as jt
import copy

#jt.set_global_seed(0)
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses, sync
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
import numpy as np
import random
import jdet
import argparse
import os
import pickle as pk
from jdet.runner import Runner 
from PIL import Image
from tqdm import tqdm
import copy

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
    init_cfg("configs/san10_pairwise.py")
    runner = Runner()
    
    # image_file = '/home/flowey/dataset/ILSVRC2012/val/n02106166/ILSVRC2012_val_00001900.JPEG'
    # image_torch_file = '/home/flowey/remote/JDet/projects/san/test_datas_san/single_image.pkl'
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

    numpy_save_dir = 'test_datas_san/models_numpy.pth'
    numpy_dict = pk.load(open(numpy_save_dir, 'rb'))
    jittor_dict = dict()
    for k, v in numpy_dict.items():
        jittor_dict[k] = jt.array(v)
    # print(len(jittor_dict.keys()))
    # print(len(runner.model.state_dict().keys()))
    # for k in runner.model.state_dict().keys():
    #     if k not in numpy_dict.keys():
    #         print("fjkdsalfjklsda")
    # return
    runner.model.load_state_dict(jittor_dict)

    runner.logger.print_log("Validating....")
    # TODO: need move eval into this function
    runner.model.eval()
    # if runner.model.is_training():
    #     runner.model.eval()
    results = []
    for batch_idx,(images,targets) in tqdm(enumerate(runner.val_dataset),total=len(runner.val_dataset)):
        # if batch_idx == 0:
        #     l, result = runner.model(images, targets, show=True)
        #     pk.dump(sync(l), open('/home/flowey/remote/JDet/projects/san/test_datas_san/layer_j.pkl', 'wb'))
        # else:
        #     continue
        result = runner.model(images,targets)
        # iimages.append(sync(images))
        # itarget.append(sync(targets))
        # results.append(sync(result))
        # layerss.append(sync(layers))
        results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])
    # save_path = 'test_datas_san/eval_results.pkl'
    # pk.dump(saved, open(save_path, 'wb'))

    eval_results = runner.val_dataset.evaluate(results,runner.work_dir,runner.epoch,logger=runner.logger)
    runner.logger.log(eval_results,iter=runner.iter)

    
if __name__ == "__main__":
    main()