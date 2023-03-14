import numpy as np
from .group import Group
from .representation import Representation, IrreducibleRepresentation
from .math_utils import psi

_cached_group_instance = None

class SO2(Group):
    def __init__(self, maximum_frequency: int):
        r"""
       Build an instance of the special orthogonal group :math:`SO(2)` which contains continuous planar rotations.
        
        A group element is a rotation :math:`r_\theta` of :math:`\theta \in [0, 2\pi)`, with group law
        :math:`r_\alpha \cdot r_\beta = r_{\alpha + \beta}`.
        
        Elements are implemented as floating point numbers :math:`\theta \in [0, 2\pi)`.
        
        .. note ::
            Since the group has infinitely many irreducible representations, it is not possible to build all of them.
            Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
            the maximum frequency of the irreps to build.
            New irreps (associated to higher frequencies) can be manually created by calling the method
            :meth:`~e2cnn.group.SO2.irrep` (see the method's documentation).
        
        Args:
            maximum_frequency (int): the maximum frequency to consider when building the irreps of the group
        
        """
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        super(SO2, self).__init__("SO(2)", True, True)
        self._maximum_frequency = maximum_frequency
        self.identity = 0.
        self._build_representations()

    def inverse(self, element: float) -> float:
        r"""
        Return the inverse element of the input element: given an angle, the method returns its opposite
        Args:
            element (float): an angle :math:`\theta`
        Returns:
            its opposite :math:`-\theta \mod 2\pi`
        """
        return (-element) % (2*np.pi)

    def combine(self, e1: float, e2: float) -> float:
        r"""
        Return the sum of the two input elements: given two angles, the method returns their sum
        Args:
            e1 (float): an angle :math:`\alpha`
            e2 (float): another angle :math:`\beta`
        Returns:
            their sum :math:`(\alpha + \beta) \mod 2\pi`
        """
        return (e1 + e2) % (2.*np.pi)

    def _build_representations(self):
        r"""
        Build the irreps for this group

        """
        # Build all the Irreducible Representations
        k = 0
        # add Trivial representation
        self.irrep(k)
        for k in range(self._maximum_frequency + 1):
            self.irrep(k)
        # Build all Representations
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**self.irreps)

    def irrep(self, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with rotational frequency :math:`k` of :math:`SO(2)`.
        Notice: the frequency has to be a non-negative integer.
        Args:
            k (int): the frequency of the irrep
        Returns:
            the corresponding irrep

        """
        assert k >= 0
        name = f"irrep_{k}"
        if name not in self.irreps:
            if k == 0:
                # Trivial representation
                irrep = lambda element, identity=np.eye(1): identity
                character = lambda e: 1
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              # trivial=True,
                                                              frequency=0
                                                              )
            else:
                # 2 dimensional Irreducible Representations
                # build the rotation matrix with rotation order 'k'
                irrep = lambda element, k=k: psi(element, k=k)
                # build the trace of this matrix
                character = lambda element, k=k: np.cos(k * element) + np.cos(k * element)
                supported_nonlinearities = ['norm', 'gated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 2, 2,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k)

        return self.irreps[name]

    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0']

    @staticmethod
    def _generator(maximum_frequency: int = 10) -> 'SO2':
        global _cached_group_instance
        if _cached_group_instance is None:
            _cached_group_instance = SO2(maximum_frequency)
        elif _cached_group_instance._maximum_frequency < maximum_frequency:
            _cached_group_instance._maximum_frequency = maximum_frequency
            _cached_group_instance._build_representations()
        return _cached_group_instance

    def is_element(self, element: float) -> bool:
        return isinstance(element, float)

    def __eq__(self, other):
        if not isinstance(other, SO2):
            return False
        else:
            return self.name == other.name and self._maximum_frequency == other._maximum_frequency


def so2_group(maximum_frequency: int = 10):
    r"""

    Builds the group :math:`SO(2)`, i.e. the group of continuous planar rotations.
    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`e2cnn.group.SO2.irrep` (see the method's documentation).
    
    You should use this factory function to build an instance of :class:`e2cnn.group.SO2`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`e2cnn.group.SO2.irrep`), this unique instance is updated with the new representations
    and, therefore, all its references will see the new representations.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`SO(2)`

    """
    return SO2._generator(maximum_frequency)

