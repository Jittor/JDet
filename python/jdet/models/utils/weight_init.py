import jittor as jt 
from jittor import init,nn 
import numpy as np

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and isinstance(module.bias,jt.Var) and module.bias is not None:
        init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            init.xavier_uniform_(module.weight, gain=gain)
        else:
            init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)  # type: ignore


def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        bias=bias,
        distribution='uniform')


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
