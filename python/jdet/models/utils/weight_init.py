import jittor as jt 
from jittor import init 

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)
