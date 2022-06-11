import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,OPTIMS
import numpy as np
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
    jt.set_global_seed(666)
    init_cfg("configs/yolo_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    resume_data = jt.load(cfg.pretrained_weights)
    model.load_parameters(resume_data['model'])
    if (cfg.parameter_groups_generator):
        params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=model.named_parameters(), model=model)
    else:
        params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)
    model.train()

    if (args.set_data):
        imagess = []
        targetss = []
        cls_losses = []
        obj_losses = []
        box_losses = []

        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
        for batch_idx,(images,targets) in enumerate(train_dataset):
            print(batch_idx)
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            # print(targets)
            # print(jdet.utils.general.sync(targets))

            losses = model(images,targets)
            l1 = losses["box_loss"].data[0]
            box_losses.append(l1)
            l2 = losses["cls_loss"].data[0]
            cls_losses.append(l2)
            l3 = losses["obj_loss"].data[0]
            obj_losses.append(l3)
            if (batch_idx > 10):
                break
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "cls_losses": cls_losses,
            "obj_losses": obj_losses,
            "box_losses": box_losses
        }
        if (not os.path.exists("test_datas_yolo")):
            os.makedirs("test_datas_yolo")
        pk.dump(data, open("test_datas_yolo/test_data.pk", "wb"))
        print(cls_losses)
        print(obj_losses)
        print(box_losses)
    else:
        data = pk.load(open("test_datas_yolo/test_data.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        cls_losses = data["cls_losses"]
        obj_losses = data["obj_losses"]
        box_losses = data["box_losses"]
        # [0.07727768, 0.08936529, 0.092872865, 0.07554055, 0.08003952, 0.08859269, 0.076792404, 0.08925526]
        # [0.01492238, 0.024641307, 0.033386815, 0.027101148, 0.019679038, 0.014860249, 0.02721583, 0.017503828]
        # [0.096790835, 0.097113855, 0.09675336, 0.0935606, 0.09338399, 0.09713681, 0.09408038, 0.094129026]
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]

            losses = model(images,targets)
            l1 = losses["cls_loss"].data[0]
            s_l1 = cls_losses[batch_idx]
            l2 = losses["obj_loss"].data[0]
            s_l2 = obj_losses[batch_idx]
            l3 = losses["box_loss"].data[0]
            s_l3 = box_losses[batch_idx]
            print(abs(l1 - s_l1) / abs(s_l1), abs(l2 - s_l2) / abs(s_l2), abs(l3 - s_l3) / abs(s_l3))
            assert(abs(l1 - s_l1) / abs(s_l1) < 1e-1)
            assert(abs(l2 - s_l2) / abs(s_l2) < 1e-1)
            assert(abs(l3 - s_l3) / abs(s_l3) < 1e-1)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
    print("success!")
if __name__ == "__main__":
    main()