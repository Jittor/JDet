
from jdet.utils.registry import OPTIMS 

from jittor import optim 

class Optimizer(object):
    def parameters(self):
        data = {}
        for k,d in self.__dict__.items():
            if k == "param_groups":
                continue
            data[k]=d 
        return data

    def load_parameters(self,data):
        if isinstance(data,dict):
            for k,d in data.items():
                if k in self.__dict__:
                    self.__dict__[k]=d 
    def cur_lr(self):
        return self.param_groups[0].get("lr",self.lr)
    
@OPTIMS.register_module()
class SGD(optim.SGD,Optimizer):
    def __init__(self,params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False,grad_clip=None):
        super(SGD,self).__init__(params, lr, momentum, weight_decay, dampening, nesterov)
        self.grad_clip = grad_clip

    def pre_step(self, loss):
        super(SGD,self).pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)
    
@OPTIMS.register_module()
class GradMutilpySGD(SGD):
    def __init__(self, **kwargs):
        super(GradMutilpySGD,self).__init__(**kwargs)

    def pre_step(self, loss):
        super(GradMutilpySGD,self).pre_step(loss)
        for pg in self.param_groups:
            if ("grad_mutilpy" in pg):
                m = pg["grad_mutilpy"]
                for p, g in zip(pg["params"], pg["grads"]):
                    g *= m

@OPTIMS.register_module()
class Adam(optim.Adam,Optimizer):
    pass 

