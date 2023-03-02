from .gspace import GSpace
from .field_type import FieldType
import math
from typing import List, Dict, Tuple
from collections import defaultdict

def regular_feature_type(gspace: GSpace, planes: int, fixparams: bool = False):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= math.sqrt(N)

    planes = planes / N
    planes = int(planes)

    return FieldType(gspace, [gspace.regular_repr] * planes)

def trivial_feature_type(gspace: GSpace, planes: int, fixparams: bool = False):
    """ build a trivial feature map with the specified number of channels"""
    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return FieldType(gspace, [gspace.trivial_repr] * planes)

def check_consecutive_numbers(list: List[int]) -> bool:    
    m = M = list[0]
    s = 0
    for l in list:
        assert l >= 0
        m = min(m, l)
        M = max(M, l)
        s += l
    S = M*(M+1)/2 - (m-1)*m/2
    return S == s

def indexes_from_labels(in_type: FieldType, labels: List[str]) -> Dict[str, Tuple[bool, List[int], List[int]]]:
    r"""
    Args:
        in_type (FieldType): the input field type
        labels (list): a list of strings long as the list :attr:'~e2cnn.nn.FieldType.representations`
                of the input :attr:`in_type`
    Returns:
    """
    assert len(labels) == len(in_type)
    
    indeces = defaultdict(lambda: [])
    fields = defaultdict(lambda: [])
    current_position = 0
    for c, (l, r) in enumerate(zip(labels, in_type.representations)):
        # append the indeces of the current field to the list corresponding to this label
        indeces[l] += list(range(current_position, current_position + r.size))
        fields[l].append(c)
        current_position += r.size
    groups = {}
    for l in labels:
        contiguous = check_consecutive_numbers(indeces[l])
        groups[l] = contiguous, fields[l], indeces[l]
    return groups

