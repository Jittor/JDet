# reference: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
from jdet.utils.registry import SCHEDULERS
import math 


@SCHEDULERS.register_module()
class WarmUpLR(object):
    """Warm LR scheduler, which is the base lr_scheduler,default we use it.
    Args:
        optimizer (Optimizer): the optimizer to optimize the model
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
    """
    def __init__(self,optimizer,
                      warmup_ratio=1.0 / 3,
                      warmup_iters=500,
                      warmup = None):
        self.optimizer = optimizer
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup = warmup
    
    def get_warmup_lr(self,lr,cur_iters):
        if self.warmup == 'constant':
            k = self.warmup_ratio
        elif self.warmup == 'linear':
            k = 1-(1 - cur_iters / self.warmup_iters) * (1 -self.warmup_ratio)
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
        return k*lr
    
    def get_lr(self,lr,steps):
        return lr 
    
    def _update_lr(self,steps,get_lr_func):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = get_lr_func(param_group.get("initial_lr",self.optimizer.lr),steps)

    def step(self,iters,epochs,by_epoch=True):
        if self.warmup is not None:
            if iters>=self.warmup_iters:
                if by_epoch:
                    self._update_lr(epochs,self.get_lr)
                else:
                    self._update_lr(iters-self.warmup_iters,self.get_lr)
            else:
                self._update_lr(iters,self.get_warmup_lr)
        else:
            if by_epoch:
                self._update_lr(epochs,self.get_lr)
            else:
                self._update_lr(iters,self.get_lr)

    def parameters(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_parameters(self,data):
        if isinstance(data,dict):
            for k,d in data.items():
                if k in self.__dict__:
                    self.__dict__[k]=d 


@SCHEDULERS.register_module()
class StepLR(WarmUpLR):
    """Step LR scheduler with min_lr clipping.
    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, milestones, gamma=0.1, min_lr=None, **kwargs):
        if isinstance(milestones, list):
            assert all([s > 0 for s in milestones])
        elif isinstance(milestones, int):
            assert milestones > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.milestones = milestones
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLR, self).__init__(**kwargs)

    def get_lr(self,base_lr, steps):
        # calculate exponential term
        if isinstance(self.milestones, int):
            exp = steps // self.milestones
        else:
            exp = len(self.milestones)
            for i, s in enumerate(self.milestones):
                if steps < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr

@SCHEDULERS.register_module()
class CosineAnnealingLR(WarmUpLR):

    def __init__(self, max_steps,min_lr=0., min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.max_steps = max_steps
        super(CosineAnnealingLR, self).__init__(**kwargs)

    def get_lr(self, base_lr,steps):
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        cos_out = math.cos(math.pi * (steps / self.max_steps)) + 1
        lr = target_lr + 0.5 * (base_lr - target_lr) * cos_out
        return lr

@SCHEDULERS.register_module()
class ExpLR(WarmUpLR):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLR, self).__init__(**kwargs)

    def get_lr(self,base_lr,steps):
        return base_lr * self.gamma**steps

@SCHEDULERS.register_module()
class PolyLR(WarmUpLR):

    def __init__(self, max_steps,power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        self.max_steps = max_steps
        super(PolyLR, self).__init__(**kwargs)

    def get_lr(self, base_lr,steps):
        coeff = (1 - steps / self.max_steps)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@SCHEDULERS.register_module()
class InvLR(WarmUpLR):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLR, self).__init__(**kwargs)

    def get_lr(self, base_lr,steps):
        return base_lr * (1 + self.gamma * steps)**(-self.power)
