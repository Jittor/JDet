import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
import models

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/retinanet_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    model.load(cfg.pretrained_weights)
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
    if (cfg.parameter_groups_generator):
        params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=model.named_parameters(), model=model)
    else:
        params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)

    std_roi_cls_losses = [1.1876051,1.213222,1.1627599,1.156189,1.2110083,1.1543972,1.164568,1.1559218,1.2017323,1.1426077,1.1537755,1.1491635]
    std_roi_loc_losses = [0.27317095,0.4550915,0.29411379,0.33598003,0.3730816,0.5620489,0.31406635,0.21403852,0.5257297,0.3200497,0.49014392,0.23513089]
    model.train()
    for batch_idx,(images,targets) in enumerate(train_dataset):
        losses = model(images,targets)
        l1 = losses["roi_cls_loss"].data[0]
        s_l1 = std_roi_cls_losses[batch_idx]
        l2 = losses["roi_loc_loss"].data[0]
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