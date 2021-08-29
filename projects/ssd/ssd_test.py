import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/ssd300_coco_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
    params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)

    std_roi_cls_losses = jt.Var([26.437416,26.762486,25.729113,26.188711,26.580881,27.949156,24.762697,25.833862,25.867805,25.638443,25.699451,26.303846])
    std_roi_loc_losses = jt.Var([2.8528306,3.5279472,2.517684,2.9012783,2.5886812,2.6759088,3.6240485,3.865853,2.7286806,7.5056534,2.933507,2.5085826])
    model.train()
    for batch_idx,(images,targets) in enumerate(train_dataset):
        losses = model(images,targets)
        
        l1 = losses["loss_cls"][0].data.reshape(1)
        s_l1 = std_roi_cls_losses[batch_idx]
        l2 = losses["loss_bbox"][0].data.reshape(1)
        s_l2 = std_roi_loc_losses[batch_idx]
        assert(abs(l1 - s_l1) / abs(s_l1) < 1e-3)
        assert(abs(l2 - s_l2) / abs(s_l2) < 1e-3)
        print(abs(l1 - s_l1) / abs(s_l1), abs(l2 - s_l2) / abs(s_l2))
        if (batch_idx > 10):
            break
        all_loss,losses = parse_losses(losses)
        optimizer.step(all_loss)
        scheduler.step(iter,0,by_epoch=True)
        iter+=1

if __name__ == "__main__":
    main()