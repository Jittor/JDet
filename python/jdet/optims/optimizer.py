from jdet.utils.registry import OPTIMS

from jittor import optim
import jittor as jt


class Optimizer(object):
    def parameters(self):
        data = {}
        for k, d in self.__dict__.items():
            if k == "param_groups":
                continue
            data[k] = d
        return data

    def load_parameters(self, data):
        if isinstance(data, dict):
            for k, d in data.items():
                if k in self.__dict__:
                    self.__dict__[k] = d

    def cur_lr(self):
        return self.param_groups[0].get("lr", self.lr)


@OPTIMS.register_module()
class SGD(optim.SGD, Optimizer):
    def __init__(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False, grad_clip=None):
        super(SGD, self).__init__(params, lr, momentum, weight_decay, dampening, nesterov)
        self.grad_clip = grad_clip

    def pre_step(self, loss, retain_graph=False):
        super(SGD, self).pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)


@OPTIMS.register_module()
class GradMutilpySGD(optim.SGD, Optimizer):
    def __init__(self, grad_clip=None, **kwargs):
        super(GradMutilpySGD, self).__init__(**kwargs)
        self.grad_clip = grad_clip

    def step(self, loss):
        if loss is not None:
            self.pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            momentum = pg.get("momentum", self.momentum)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            dampening = pg.get("dampening", self.dampening)
            nesterov = pg.get("nesterov", self.nesterov)

            m = pg.get("grad_mutilpy", 1)
            # optimize main body
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad(): continue
                dp = p * weight_decay + g * m
                v.update(momentum * v + dp * (1 - dampening))
                if nesterov:
                    p.update(p - (dp + momentum * v) * lr)
                else:
                    p.update(p - v * lr)
        self.zero_grad()


@OPTIMS.register_module()
class Adam(optim.Adam, Optimizer):
    pass


@OPTIMS.register_module()
class AdamW(optim.AdamW, Optimizer):
    pass