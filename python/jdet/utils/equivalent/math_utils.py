import numpy as np
import math
from typing import Union

def psi(theta: float, k: int = 1, gamma: float = 0.):
    r"""
    Rotation matrix corresponding to the angle :math:`k \theta + \gamma`.
    """
    x = k * theta + gamma
    c, s = np.cos(x), np.sin(x)
    return np.array(([
        [c, -s],
        [s,  c],
    ]))


def psichi(theta: Union[np.ndarray, float], s: int, k: int = 1, gamma: float = 0., out: np.ndarray = None) -> np.ndarray:
    # equal to the matrix multiplication of psi(theta, k, gamma) and chi(s) along the first 2 axis
    
    if isinstance(theta, float):
        theta = np.array(theta)
    
    s = -1 * (s > 0) + (s <= 0)
    
    s = np.array(s, copy=False).reshape(-1, 1)
    k = np.array(k, copy=False).reshape(-1, 1)
    gamma = np.array(gamma, copy=False).reshape(-1, 1)
    theta = theta.reshape(1, -1)
    
    x = k * theta + gamma
    
    cos, sin = np.cos(x), np.sin(x)
    
    if out is None:
        out = np.empty((2, 2, x.shape[0], x.shape[-1]))
    
    out[0, 0, ...] = cos
    out[0, 1, ...] = -s*sin
    out[1, 0, ...] = sin
    out[1, 1, ...] = s*cos
    
    return out

def offset_iterator(base_frequency, N, maximum_offset=None, maximum_frequency=None, non_negative: bool = False):
    if N < 0:
        # assert maximum_offset == 0
        if maximum_frequency is not None and math.fabs(base_frequency) <= maximum_frequency:
            yield 0
    else:
        assert maximum_frequency is not None or maximum_offset is not None
        
        if non_negative:
            minimum_frequency = 0
        else:
            minimum_frequency = -maximum_frequency if maximum_frequency is not None else None

        def round(x):
            if x > 0:
                return int(math.floor(x))
            else:
                return int(math.ceil(x))
            
        if maximum_frequency is not None:
            min_offset = (minimum_frequency - base_frequency) / N
            max_offset = (maximum_frequency - base_frequency) / N
        else:
            min_offset = -10000
            max_offset = 10000
            
        if maximum_offset is not None:
            min_offset = max(min_offset, -maximum_offset)
            max_offset = min(max_offset, maximum_offset)
        
        min_offset = math.ceil(min_offset)
        max_offset = math.floor(max_offset)
        
        for j in range(min_offset, max_offset+1):
            yield j
