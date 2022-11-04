import numpy as np
from typing import List, Union

from ..representation import Representation
from ..cyclic_group import CyclicGroup
from .basis import KernelBasis, GaussianRadialProfile, PolarBasis
from .steerable_basis import SteerableKernelBasis
from .irreps_basis import R2DiscreteRotationsSolution

# def kernels_O2_act_R2(in_repr: Representation, out_repr: Representation,
#                       radii: List[float],
#                       sigma: Union[List[float], float],
#                       axis: float = np.pi / 2) -> KernelBasis:
#     r"""

#     Builds a basis for convolutional kernels equivariant to reflections and continuous rotations, modeled by the
#     group :math:`O(2)`.
#     ``in_repr`` and ``out_repr`` need to be :class:`~e2cnn.group.Representation` s of :class:`~e2cnn.group.O2`.

#     Because the equivariance constraints allow any choice of radial profile, we use a
#     :class:`~e2cnn.kernels.GaussianRadialProfile`.
#     ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
#     of these rings (see :class:`~e2cnn.kernels.GaussianRadialProfile`).

#     Because :math:`O(2)` contains all rotations, the reflection element of the group can be associated to any reflection
#     axis. Reflections along other axes can be obtained by composition with rotations.
#     However, a choice of this axis is required to align the basis with respect to the action of the group.

#     Args:
#         in_repr (Representation): the representation specifying the transformation of the input feature field
#         out_repr (Representation): the representation specifying the transformation of the output feature field
#         radii (list): radii of the rings defining the basis for the radial profile
#         sigma (list or float): widths of the rings defining the basis for the radial profile
#         axis (float, optional): angle of the axis of the reflection element

#     """
#     assert in_repr.group == out_repr.group
    
#     group = in_repr.group
#     assert isinstance(group, O2)
    
#     angular_basis = SteerableKernelBasis(R2FlipsContinuousRotationsSolution, in_repr, out_repr, axis=axis)
    
#     radial_profile = GaussianRadialProfile(radii, sigma)
    
#     return PolarBasis(radial_profile, angular_basis)


def kernels_CN_act_R2(in_repr: Representation, out_repr: Representation,
                      radii: List[float],
                      sigma: Union[List[float], float],
                      max_frequency: int = None,
                      max_offset: int = None) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to :math:`N` discrete rotations, modeled by
    the group :math:`C_N`.
    ``in_repr`` and ``out_repr`` need to be :class:`~e2cnn.group.Representation` s of :class:`~e2cnn.group.CyclicGroup`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~e2cnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~e2cnn.kernels.GaussianRadialProfile`).
    
    The analytical angular solutions of kernel constraints belong to an infinite dimensional space and so can be
    expressed in terms of infinitely many basis elements, each associated with one unique frequency. Because the kernels
    are then sampled on a finite number of points (e.g. the cells of a grid), only low-frequency solutions needs to be
    considered. This enables us to build a finite dimensional basis containing only a finite subset of all analytical
    solutions. ``max_frequency`` is an integer controlling the highest frequency sampled in the basis.
    
    Frequencies also appear in a basis with a period of :math:`N`, i.e. if the basis contains an element with frequency
    :math:`k`, then it also contains an element with frequency :math:`k + N`.
    In the analytical solutions shown in Table 11 `here <https://arxiv.org/abs/1911.08251>`_, each solution has a
    parameter :math:`t` or :math:`\hat{t}`.
    ``max_offset`` defines the maximum absolute value of these two numbers.
    
    Either ``max_frequency`` or ``max_offset`` must be specified.
    

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        max_frequency (int): maximum frequency of the basis
        max_offset (int): maximum offset in the frequencies of the basis

    """
    
    assert in_repr.group == out_repr.group
    
    group = in_repr.group

    assert isinstance(group, CyclicGroup)
    
    angular_basis = SteerableKernelBasis(R2DiscreteRotationsSolution, in_repr, out_repr,
                                         max_frequency=max_frequency,
                                         max_offset=max_offset)

    radial_profile = GaussianRadialProfile(radii, sigma)
    
    return PolarBasis(radial_profile, angular_basis)
