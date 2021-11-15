import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
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
    init_cfg("configs/gliding_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())

    model.train()
    
    if (args.set_data):
        os.makedirs("test_datas_gliding",exist_ok=True)
        jt.save(model.state_dict(), "test_datas_gliding/model.pk")
        imagess = []
        targetss = []
        correct_loss = []
        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        for batch_idx,(images,targets) in enumerate(train_dataset):
            if (batch_idx > 10):
                break
            print(batch_idx)
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            correct_loss.append(all_loss.item())
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "correct_loss": correct_loss,
        }
        if (not os.path.exists("test_datas_gliding")):
            os.makedirs("test_datas_gliding")
        pk.dump(data, open("test_datas_gliding/test_gliding.pk", "wb"))
        print(correct_loss)

    else:

        model.load_parameters(jt.load("test_datas_gliding/model.pk"))
        data = pk.load(open("test_datas_gliding/test_gliding.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        correct_loss = data["correct_loss"]
        # correct_loss = [{'gliding_cls_loss': 2.8607935905456543, 'gliding_bbox_loss': 0.004707167390733957, 'gliding_fix_loss': 0.010641977190971375, 'gliding_ratio_loss': 0.03824465721845627, 'loss_rpn_cls': 0.727114737033844, 'loss_rpn_bbox': 0.03236636891961098, 'total_loss': 3.673868417739868}, {'gliding_cls_loss': 1.4579474925994873, 'gliding_bbox_loss': 0.021469490602612495, 'gliding_fix_loss': 0.021563515067100525, 'gliding_ratio_loss': 0.07166668772697449, 'loss_rpn_cls': 0.6899648904800415, 'loss_rpn_bbox': 0.14480000734329224, 'total_loss': 2.407412052154541}, {'gliding_cls_loss': 1.2834120988845825, 'gliding_bbox_loss': 0.10129604488611221, 'gliding_fix_loss': 0.15012003481388092, 'gliding_ratio_loss': 0.5383837223052979, 'loss_rpn_cls': 0.6802653074264526, 'loss_rpn_bbox': 0.18992501497268677, 'total_loss': 2.943402051925659}, {'gliding_cls_loss': 0.2936493158340454, 'gliding_bbox_loss': 0.06922154873609543, 'gliding_fix_loss': 0.009447498247027397, 'gliding_ratio_loss': 0.005887447856366634, 'loss_rpn_cls': 0.5001091957092285, 'loss_rpn_bbox': 0.06825030595064163, 'total_loss': 0.9465652704238892}, {'gliding_cls_loss': 0.4967324435710907, 'gliding_bbox_loss': 0.07386171072721481, 'gliding_fix_loss': 0.02249765582382679, 'gliding_ratio_loss': 0.055555325001478195, 'loss_rpn_cls': 0.46851035952568054, 'loss_rpn_bbox': 0.1712641716003418, 'total_loss': 1.288421630859375}, {'gliding_cls_loss': 1.2238610982894897, 'gliding_bbox_loss': 0.15869204699993134, 'gliding_fix_loss': 0.07835061103105545, 'gliding_ratio_loss': 0.10634005069732666, 'loss_rpn_cls': 0.5587880611419678, 'loss_rpn_bbox': 0.1531504988670349, 'total_loss': 2.2791824340820312}, {'gliding_cls_loss': 0.183293879032135, 'gliding_bbox_loss': 0.05720524862408638, 'gliding_fix_loss': 0.02445354126393795, 'gliding_ratio_loss': 0.08038179576396942, 'loss_rpn_cls': 0.4437496066093445, 'loss_rpn_bbox': 0.018438303843140602, 'total_loss': 0.8075223565101624}, {'gliding_cls_loss': 0.10430146008729935, 'gliding_bbox_loss': 0.0209798626601696, 'gliding_fix_loss': 0.009506018832325935, 'gliding_ratio_loss': 0.023182889446616173, 'loss_rpn_cls': 0.3377474248409271, 'loss_rpn_bbox': 0.04412488266825676, 'total_loss': 0.5398425459861755}, {'gliding_cls_loss': 0.12577053904533386, 'gliding_bbox_loss': 0.003616021480411291, 'gliding_fix_loss': 0.017107533290982246, 'gliding_ratio_loss': 0.041523586958646774, 'loss_rpn_cls': 0.3998869061470032, 'loss_rpn_bbox': 0.03059343248605728, 'total_loss': 0.6184980273246765}, {'gliding_cls_loss': 0.03747999668121338, 'gliding_bbox_loss': 2.9481586238944146e-07, 'gliding_fix_loss': 0.002199558075517416, 'gliding_ratio_loss': 0.0017057707300409675, 'loss_rpn_cls': 0.1920843869447708, 'loss_rpn_bbox': 0.025726553052663803, 'total_loss': 0.25919654965400696}, {'gliding_cls_loss': 0.5704915523529053, 'gliding_bbox_loss': 0.11582531034946442, 'gliding_fix_loss': 0.051260657608509064, 'gliding_ratio_loss': 0.13672445714473724, 'loss_rpn_cls': 0.399156391620636, 'loss_rpn_bbox': 0.0785030648112297, 'total_loss': 1.351961374282837}]
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]
            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            l = all_loss.item()
            c_l = correct_loss[batch_idx]
            err_rate = abs(c_l-l)/min(c_l,l)
            print(f"correct loss is {c_l:.4f}, runtime loss is {l:.4f}, err rate is {err_rate*100:.2f}%")
            assert err_rate<2e-1,"LOSS is not correct, please check it"

        print(f"Loss is correct with err_rate<{2e-1}")

    print("success!")
    
if __name__ == "__main__":
    main()