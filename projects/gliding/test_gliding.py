import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
import numpy as np 
import random
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS

def sync_item(losses):
    return {k:d.item() for k,d in losses.items()}

def check(losses,c_losses):
    for k,l in losses.items():
        c_l = c_losses[k]
        err_rate = abs(c_l-l)
        print(f"{k} correct loss is {c_l:.4f}, runtime loss is {l:.4f}, err rate is {err_rate*100:.2f}%")
        assert err_rate<3e-2,f"{k} is not correct with 1e-3, please check it"
    print("<-------------------------------------->")

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    np.random.seed(666)
    random.seed(666)
    init_cfg("configs/gliding_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())
    model.train()

    correct_loss = [{'gliding_cls_loss': 2.75551438331604, 'gliding_bbox_loss': 0.019278723746538162, 'gliding_fix_loss': 0.022597042843699455, 'gliding_ratio_loss': 0.004687775857746601, 'loss_rpn_cls': 0.8330771327018738, 'loss_rpn_bbox': 0.06085281819105148, 'total_loss': 3.6960079669952393},
                    {'gliding_cls_loss': 2.13930606842041, 'gliding_bbox_loss': 0.04994218051433563, 'gliding_fix_loss': 0.047657985240221024, 'gliding_ratio_loss': 0.008912277407944202, 'loss_rpn_cls': 0.7671789526939392, 'loss_rpn_bbox': 0.24532891809940338, 'total_loss': 3.258326530456543},
                    {'gliding_cls_loss': 1.3566994667053223, 'gliding_bbox_loss': 0.19929386675357819, 'gliding_fix_loss': 0.18524281680583954, 'gliding_ratio_loss': 0.03633378818631172, 'loss_rpn_cls': 0.6917773485183716, 'loss_rpn_bbox': 0.3835605978965759, 'total_loss': 2.85290789604187}, 
                    {'gliding_cls_loss': 0.5115861296653748, 'gliding_bbox_loss': 0.02242315374314785, 'gliding_fix_loss': 0.015583056956529617, 'gliding_ratio_loss': 0.0010958444327116013, 'loss_rpn_cls': 0.6516796350479126, 'loss_rpn_bbox': 0.11049643158912659, 'total_loss': 1.3128641843795776},
                    {'gliding_cls_loss': 0.2530573308467865, 'gliding_bbox_loss': 0.03870411962270737, 'gliding_fix_loss': 0.028401846066117287, 'gliding_ratio_loss': 0.0026811466086655855, 'loss_rpn_cls': 0.5703112483024597, 'loss_rpn_bbox': 0.19726413488388062, 'total_loss': 1.0904197692871094},
                    {'gliding_cls_loss': 0.45836639404296875, 'gliding_bbox_loss': 0.18862147629261017, 'gliding_fix_loss': 0.12159030884504318, 'gliding_ratio_loss': 0.01177958119660616, 'loss_rpn_cls': 0.6520779728889465, 'loss_rpn_bbox': 0.29513972997665405, 'total_loss': 1.7275755405426025},
                    {'gliding_cls_loss': 0.08597975969314575, 'gliding_bbox_loss': 0.020654886960983276, 'gliding_fix_loss': 0.022463221102952957, 'gliding_ratio_loss': 0.0038461857475340366, 'loss_rpn_cls': 0.45672693848609924, 'loss_rpn_bbox': 0.048490557819604874, 'total_loss': 0.6381615996360779}, 
                    {'gliding_cls_loss': 0.06654457747936249, 'gliding_bbox_loss': 0.017759224399924278, 'gliding_fix_loss': 0.013898524455726147, 'gliding_ratio_loss': 0.002535144565626979, 'loss_rpn_cls': 0.3815120756626129, 'loss_rpn_bbox': 0.0815621092915535, 'total_loss': 0.5638116598129272}, 
                    {'gliding_cls_loss': 0.16983596980571747, 'gliding_bbox_loss': 0.05806876718997955, 'gliding_fix_loss': 0.03988276422023773, 'gliding_ratio_loss': 0.006142245605587959, 'loss_rpn_cls': 0.4762706756591797, 'loss_rpn_bbox': 0.08805757015943527, 'total_loss': 0.8382580280303955}, 
                    {'gliding_cls_loss': 0.07118498533964157, 'gliding_bbox_loss': 0.025214344263076782, 'gliding_fix_loss': 0.006861195433884859, 'gliding_ratio_loss': 0.00046937624574638903, 'loss_rpn_cls': 0.2955016493797302, 'loss_rpn_bbox': 0.04748834669589996, 'total_loss': 0.4467198848724365}, 
                    {'gliding_cls_loss': 0.26269587874412537, 'gliding_bbox_loss': 0.05339872092008591, 'gliding_fix_loss': 0.05363383889198303, 'gliding_ratio_loss': 0.009859815239906311, 'loss_rpn_cls': 0.3878805935382843, 'loss_rpn_bbox': 0.14190295338630676, 'total_loss': 0.9093717336654663}]
    c_losses = []
    for batch_idx,(images,targets) in enumerate(train_dataset):
        losses = model(images,targets)
        if (batch_idx > 10):
            break
        all_loss,losses = parse_losses(losses)
        optimizer.step(all_loss)
        losses["total_loss"] = all_loss
        losses = sync_item(losses)
        c_losses.append(losses)
        check(c_losses[batch_idx],correct_loss[batch_idx])

    # print(c_losses)
    print(f"Loss is correct with err_rate<{2e-2}")
    
if __name__ == "__main__":
    main()
