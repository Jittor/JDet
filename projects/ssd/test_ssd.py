import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
import argparse
import os
import pickle as pk
import jdet

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/ssd300_coco_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)

    model.train()

    if (args.set_data):
        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
        imagess = []
        targetss = []
        std_roi_cls_losses = []
        std_roi_loc_losses = []
        for batch_idx,(images,targets) in enumerate(train_dataset):
            print(batch_idx)
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            losses = model(images,targets)
            
            l1 = losses["loss_cls"][0].data.reshape(1).item()
            l2 = losses["loss_bbox"][0].data.reshape(1).item()
            std_roi_cls_losses.append(l1)
            std_roi_loc_losses.append(l2)
            if (batch_idx > 10):
                break
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "std_roi_cls_losses": std_roi_cls_losses,
            "std_roi_loc_losses": std_roi_loc_losses,
        }
        if (not os.path.exists("test_datas_ssd")):
            os.makedirs("test_datas_ssd")
        pk.dump(data, open("test_datas_ssd/test_data.pk", "wb"))
        print(std_roi_cls_losses)
        print(std_roi_loc_losses)
    else:
        data = pk.load(open("test_datas_ssd/test_data.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        std_roi_cls_losses = data["std_roi_cls_losses"]
        std_roi_loc_losses = data["std_roi_loc_losses"]
        # std_roi_cls_losses = jt.Var([26.437416,26.762486,25.729113,26.188711,26.580881,27.949156,24.762697,25.833862,25.867805,25.638443,25.699451,26.303846])
        # std_roi_loc_losses = jt.Var([2.8528306,3.5279472,2.517684,2.9012783,2.5886812,2.6759088,3.6240485,3.865853,2.7286806,7.5056534,2.933507,2.5085826])
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]
            
            losses = model(images,targets)            
            l1 = losses["loss_cls"][0].data.reshape(1)
            s_l1 = std_roi_cls_losses[batch_idx]
            l2 = losses["loss_bbox"][0].data.reshape(1)
            s_l2 = std_roi_loc_losses[batch_idx]
            print(abs(l1 - s_l1) / abs(s_l1), abs(l2 - s_l2) / abs(s_l2))
            assert(abs(l1 - s_l1) / abs(s_l1) < 1e-3)
            assert(abs(l2 - s_l2) / abs(s_l2) < 1e-3)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
    print("success!")

if __name__ == "__main__":
    main()