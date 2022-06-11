import jittor as jt 
from jittor import nn 
from jdet.utils.registry import MODELS
from jittor import init, Module, Var

@MODELS.register_module()
def YoloParameterGroupsGenerator(model=None, weight_decay=-1, batch_size=64, named_params=None):
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {weight_decay}")
    normal_group = {'params': []}
    weight_group = {'params': [], 'weight_decay': weight_decay} if weight_decay > 0 else {'params': []}
    bias_group = {'params': []} 
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, jt.Var):
            bias_group['params'].append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm):
            normal_group['params'].append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, jt.Var):
            weight_group['params'].append(v.weight)  # apply decay

    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(bias_group['params']), len(weight_group['params']), len(normal_group['params'])))
    return [normal_group, weight_group, bias_group]