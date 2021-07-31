import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/s2anet_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())

    model.train()
    correct_loss =[4.851482391357422, 4.919872760772705, 3.1842665672302246, 3.716217041015625, 4.287736415863037, 
         3.794440269470215, 3.7207441329956055, 3.743844509124756, 4.571873664855957, 5.585651397705078, 3.2345163822174072]
    for batch_idx,(images,targets) in enumerate(train_dataset):
        losses = model(images,targets)
        if (batch_idx > 10):
            break
        all_loss,losses = parse_losses(losses)
        optimizer.step(all_loss)
        l = all_loss.item()
        c_l = correct_loss[batch_idx]
        err_rate = abs(c_l-l)/min(c_l,l)
        print(f"correct loss is {c_l:.4f}, runtime loss is {l:.4f}, err rate is {err_rate*100:.2f}%")
        assert err_rate<1e-3,"LOSS is not correct, please check it"
    print(f"Loss is correct with err_rate<{1e-3}")
    
if __name__ == "__main__":
    main()