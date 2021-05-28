from jdet.utils.registry import OPTIMS
from jittor.lr_scheduler import CosineAnnealingLR

OPTIMS.register_module(module=CosineAnnealingLR)


class WarmUpLR(object):
    def __init__(self,optimizer,
                      warmup_factor=1.0 / 3,
                      warmup_iters=500,
                        warmup_method="linear",):
        self.optimizer = optimizer
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
    
    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
