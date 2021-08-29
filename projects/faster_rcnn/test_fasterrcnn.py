import jittor as jt
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
    init_cfg("configs/faster_rcnn_obb_r50_fpn_1x_dota_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())

    model.train()


    if (args.set_data):
        imagess = []
        targetss = []
        correct_loss = []

        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        for batch_idx,(images,targets) in enumerate(train_dataset):
            print(batch_idx)
            if (batch_idx > 10):
                break
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            losses = model(images,targets)
            all_loss, losses = parse_losses(losses)
            optimizer.step(all_loss)

            [losses['loss_rpn_cls'].item(), losses['loss_rpn_bbox'].item(), losses['rbbox_loss_cls'].item(), losses['rbbox_acc'].item(), losses['rbbox_loss_bbox'].item(), all_loss.item()]
            correct_loss.append(all_loss.item())
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "correct_loss": correct_loss,
        }
        if (not os.path.exists("test_datas")):
            os.makedirs("test_datas")
        pk.dump(data, open("test_datas/test_data.pk", "wb"))
        print(correct_loss)
    else:
        data = pk.load(open("test_datas/test_data.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        for batch_idx in range(len(targetss)):
            targetss[batch_idx]["gt_masks"] = data["targetss"][batch_idx]["gt_masks"]
        correct_loss = data["correct_loss"]
        thr = 0.2
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]

            losses = model(images,targets)
            all_loss, losses = parse_losses(losses)
            optimizer.step(all_loss)
            l = [losses['loss_rpn_cls'].item(), losses['loss_rpn_bbox'].item(), losses['rbbox_loss_cls'].item(), losses['rbbox_acc'].item(), losses['rbbox_loss_bbox'].item(), all_loss.item()]
            l = l[-1]
            c_l = correct_loss[batch_idx]
            err_rate = float(abs(c_l - l)/np.minimum(c_l, l))
            print(f"correct loss is {float(c_l):.4f}, runtime loss is {float(l):.4f}, err rate is {err_rate*100:.2f}%")
            assert err_rate<thr,f"LOSS is not correct, please check it"
        print(f"Loss is correct with err_rate<{thr}")
    
if __name__ == "__main__":
    main()