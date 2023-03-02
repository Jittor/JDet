import numpy as np
from .group import Group
from .representation import Representation, IrreducibleRepresentation
from .math_utils import psi

_cached_group_instances = {}

class CyclicGroup(Group):
    def __init__(self, N: int):
        r"""
        Build an instance of the cyclic group :math:`C_N` which contains :math:`N` discrete planar rotations.

        The group elements are :math:`\{e, r, r^2, r^3, \dots, r^{N-1}\}`, with group law
        :math:`r^a \cdot r^b = r^{\ a + b \!\! \mod \!\! N \ }`.
        The cyclic group :math:`C_N` is isomorphic to the integers *modulo* ``N``.
        For this reason, elements are stored as the integers between :math:`0` and :math:`N-1`, where the :math:`k`-th
        element can also be interpreted as the discrete rotation by :math:`k\frac{2\pi}{N}`.
        
        Args:
            N (int): order of the group
            
        """
        assert (isinstance(N, int) and N > 0)
        super(CyclicGroup, self).__init__("C%d" % N, False, True)
        self.elements = list(range(N))
        self.elements_names = ['e'] + ['r%d' % i for i in range(1, N)]
        self.identity = 0
        self._build_representations()

    def _build_representations(self):
        r"""
        Build the irreps and the regular representation for this group
        """
        N = self.order()
        # Build all the Irreducible Representations
        for k in range(0, int(N // 2) + 1):
            self.irrep(k)
        # Build all Representations
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**self.irreps)
        # build the regular representation
        self.representations['regular'] = self.regular_representation
        self.representations['regular'].supported_nonlinearities.add('vectorfield')

    def irrep(self, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep of frequency ``k`` of the current cyclic group.
        The frequency has to be a non-negative integer in :math:`\{0, \dots, \left \lfloor N/2 \right \rfloor \}`,
        where :math:`N` is the order of the group.
        
        Args:
            k (int): the frequency of the representation

        Returns:
            the corresponding irrep

        """
        assert 0 <= k <= self.order()//2
        name = f"irrep_{k}"
        if name not in self.irreps:
            n = self.order()
            base_angle = 2.0 * np.pi / n
            
            if k == 0:
                # Trivial representation
            
                irrep = lambda element, identity=np.eye(1): identity
                character = lambda e: 1
                supported_nonlinearities = ['pointwise', 'gate', 'norm', 'gated', 'concatenated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              # character=character,
                                                              # trivial=True,
                                                              frequency=k)
            elif n % 2 == 0 and k == int(n/2):
                # 1 dimensional Irreducible representation (only for even order groups)
                irrep = lambda element, k=k, base_angle=base_angle: np.array([[np.cos(k * element * base_angle)]])
                supported_nonlinearities = ['norm', 'gated', 'concatenated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              frequency=k)
            else:
                # 2 dimensional Irreducible Representations
                
                # build the rotation matrix with rotation frequency 'frequency'
                irrep = lambda element, k=k, base_angle=base_angle: psi(element * base_angle, k=k)
            
                supported_nonlinearities = ['norm', 'gated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 2, 2,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              frequency=k)
        return self.irreps[name]

    def combine(self, e1: int, e2: int) -> int:
        r"""
        Return the composition of the two input elements.
        Given two integers :math:`a` and :math:`b` representing the elements :math:`r^a` and :math:`r^b`, the method
        returns the integer :math:`a + b \mod N` representing the element :math:`r^{a + b \mod N}`.
        

        Args:
            e1 (int): a group element :math:`r^a`
            e2 (int): another group element :math:`r^a`

        Returns:
            their composition :math:`r^{a+b \mod N}`
            
        """
        return (e1 + e2) % self.order()

    def inverse(self, element: int) -> int:
        r"""
        Return the inverse element :math:`r^{-j \mod N}` of the input element :math:`r^j`, specified by the input
        integer :math:`j` (``element``)
        
        Args:
            element (int): a group element :math:`r^j`

        Returns:
            its opposite :math:`r^{-j \mod N}`
            
        """
        return (-element) % self.order()

    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0']

    @staticmethod
    def _generator(N: int) -> 'CyclicGroup':
        global _cached_group_instances
        if N not in _cached_group_instances:
            _cached_group_instances[N] = CyclicGroup(N)
        return _cached_group_instances[N]

    def is_element(self, element: int) -> bool:
        if isinstance(element, int):
            return 0 <= element < self.order()
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, CyclicGroup):
            return False
        else:
            return self.name == other.name and self.order() == other.order()



def cyclic_group(N: int):
    r"""
    Builds a cyclic group :math:`C_N`of order ``N``, i.e. the group of ``N`` discrete planar rotations.
    
    You should use this factory function to build an instance of :class:`e2cnn.group.CyclicGroup`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~e2cnn.group.Group.quotient_representation`), this unique instance is updated with
    the new representations and, therefore, all its references will see the new representations.
    Args:
        N (int): number of discrete rotations in the group
    Returns:
        the cyclic group of order ``N``

    """
    return CyclicGroup._generator(N)
