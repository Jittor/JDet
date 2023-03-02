from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List
from .group import Group
from .cyclic_group import CyclicGroup, cyclic_group
from .so2_group import SO2, so2_group
from .representation import Representation
from .kernels import KernelBasis, kernels_CN_act_R2

class GSpace(ABC):
    def __init__(self, fibergroup: Group, dimensionality: int, name: str):
        r"""
        Abstract class for G-spaces.
        
        A ``GSpace`` describes the space where a signal lives (e.g. :math:`\R^2` for planar images) and its symmetries
        (e.g. rotations or reflections).
        As an `Euclidean` base space is assumed, a G-space is fully specified by the ``dimensionality`` of the space
        and a choice of origin-preserving symmetry group (``fibergroup``).
        
        .. note ::
            Mathematically, this class describes a *Principal Bundle*
            :math:`\pi : (\R^D, +) \rtimes G \to \mathbb{R}^D, tg \mapsto tG`,
            with the Euclidean space :math:`\mathbb{R}^D` (where :math:`D` is the ``dimensionality``) as `base space`
            and :math:`G` as `fiber group` (``fibergroup``).
            For more details on this interpretation we refer to
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.
        Args:
            fibergroup (Group): the fiber group
            dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            name (str): an identification name
        
        Attributes:
            ~.fibergroup (Group): the fiber group
            ~.dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            ~.name (str): an identification name
            ~.basespace (str): the name of the space whose symmetries are modeled. It is an Euclidean space :math:`\R^D`.
        """

        self.name = name
        self.dimensionality = dimensionality
        self.fibergroup = fibergroup
        self.basespace = f"R^{self.dimensionality}"

    @property
    def trivial_repr(self) -> Representation:
        r"""
        The trivial representation of the fiber group of this space.
        """
        return self.fibergroup.trivial_representation

    @property
    def regular_repr(self) -> Representation:
        r"""
        The regular representation of the fiber group of this space.

        .. seealso::

            :attr:`e2cnn.group.Group.regular_representation`

        """
        return self.fibergroup.regular_representation

    @abstractmethod
    def build_kernel_basis(self,
                           in_repr: Representation,
                           out_repr: Representation,
                           **kwargs) -> KernelBasis:
        r"""

        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~e2cnn.gspaces.GSpace`.

        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^D
        
        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.
        
        This method relies on the functionalities implemented in :mod:`e2cnn.kernels` and returns an instance of
        :class:`~e2cnn.kernels.KernelBasis`.


        Args:
            in_repr (Representation): the representation associated with the input field
            out_repr (Representation): the representation associated with the output field
            **kwargs: additional keyword arguments for the equivariance contraint solver

        Returns:

            a basis for space of equivariant convolutional kernels


        """
        pass

class GeneralOnR2(GSpace):
    def __init__(self, fibergroup: Group, name: str):
        r"""
        Abstract class for the G-spaces which define the symmetries of the plane :math:`\R^2`.
        Args:
            fibergroup (Group): group of origin-preserving symmetries (fiber group)
            name (str): identification name
        """
        super(GeneralOnR2, self).__init__(fibergroup, 2, name)
        
        # in order to not recompute the basis for the same intertwiner as many times as it appears, we store the basis
        # in these dictionaries the first time we compute it
        
        # Store the computed intertwiners between irreps
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_irrep, output_irrep) pairs to the corresponding basis
        self._irreps_intertwiners_basis_memory = defaultdict(lambda: dict())

        # Store the computed intertwiners between general representations
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_repr, output_repr) pairs to the corresponding basis
        self._fields_intertwiners_basis_memory = defaultdict(dict)

    @abstractmethod
    def _basis_generator(self,
                         in_repr: Representation,
                         out_repr: Representation,
                         rings: List[float],
                         sigma: List[float],
                         **kwargs):
        pass

    def build_kernel_basis(self,
                           in_repr: Representation,
                           out_repr: Representation,
                           sigma: Union[float, List[float]],
                           rings: List[float],
                           **kwargs) -> KernelBasis:
        r"""
        
        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~e2cnn.gspaces.GSpace`.
        
        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^2
        
        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.
        
        
        Because the equivariance constraints only restrict the angular part of the kernels, any radial profile is
        permitted.
        The basis for the radial profile used here contains rings with different radii (``rings``)
        associated with (possibly different) widths (``sigma``).
        A ring is implemented as a Gaussian function over the radial component, centered at one radius
        (see also :class:`~e2cnn.kernels.GaussianRadialProfile`).
        
        .. note ::
            This method is a wrapper for the functions building the bases which are defined in :doc:`e2cnn.kernels`:
            
            - :meth:`e2cnn.kernels.kernels_O2_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_SO2_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_DN_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_CN_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_Flip_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_Trivial_act_R2`
            
            
        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            sigma (list or float): parameters controlling the width of each ring of the radial profile.
                    If only one scalar is passed, it is used for all rings
            rings (list): radii of the rings defining the radial profile
            **kwargs: Group-specific keywords arguments for ``_basis_generator`` method

        Returns:
            the analytical basis
        
        """
        
        assert isinstance(in_repr, Representation)
        assert isinstance(out_repr, Representation)
        
        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup
        
        if isinstance(sigma, float):
            sigma = [sigma] * len(rings)

        assert all([s > 0. for s in sigma])
        assert len(sigma) == len(rings)
        
        # build the key
        key = dict(**kwargs)
        key["sigma"] = tuple(sigma)
        key["rings"] = tuple(rings)
        key = tuple(sorted(key.items()))

        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory[key]:
            # TODO - we could use a flag in the args to choose whether to store it or not
            
            basis = self._basis_generator(in_repr, out_repr, rings, sigma, **kwargs)
       
            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)] = basis

        # return the dictionary with all the basis built for this filter size
        return self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)]

class Rot2dOnR2(GeneralOnR2):
    def __init__(self, N: int = None, maximum_frequency: int = None, fibergroup: Group = None):
        r"""

        Describes rotation symmetries of the plane :math:`\R^2`.

        If ``N > 1``, the class models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
        (:class:`~e2cnn.group.CyclicGroup`).
        Otherwise, if ``N=-1``, the class models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
        In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
        :class:`~e2cnn.group.SO2` (see its documentation for more details)

        Args:
            N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
            maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``
            fibergroup (Group, optional): use an already existing instance of the symmetry group.
                   In that case, the other parameters should not be provided.

        """
        
        assert N is not None or fibergroup is not None, "Error! Either use the parameter `N` or the parameter `group`!"
    
        if fibergroup is not None:
            assert isinstance(fibergroup, CyclicGroup) or isinstance(fibergroup, SO2)
            assert maximum_frequency is None, "Maximum Frequency can't be set when the group is already provided in input"
            N = fibergroup.order()
            
        assert isinstance(N, int)
        
        if N > 1:
            assert maximum_frequency is None, "Maximum Frequency can't be set for finite cyclic groups"
            name = '{}-Rotations'.format(N)
        elif N == -1:
            name = 'Continuous-Rotations'
        else:
            raise ValueError(f'Error! "N" has to be an integer greater than 1 or -1, but got {N}')

        if fibergroup is None:
            if N > 1:
                fibergroup = cyclic_group(N)
            elif N == -1:
                fibergroup = so2_group(maximum_frequency)

        super(Rot2dOnR2, self).__init__(fibergroup, name)

    def _basis_generator(self,
                         in_repr: Representation,
                         out_repr: Representation,
                         rings: List[float],
                         sigma: List[float],
                         **kwargs,
                         ) -> KernelBasis:
        r"""
        Method that builds the analitical basis that spans the space of equivariant filters which
        are intertwiners between the representations induced from the representation ``in_repr`` and ``out_repr``.

        If this :class:`~e2cnn.gspaces.GSpace` includes only a discrete number of rotations (``N > 1``), either ``maximum_frequency``
        or ``maximum_offset``  must be set in the keywords arguments.

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            rings (list): radii of the rings where to sample the bases
            sigma (list): parameters controlling the width of each ring where the bases are sampled.

        Keyword Args:
            maximum_frequency (int): the maximum frequency allowed in the basis vectors
            maximum_offset (int): the maximum frequencies offset for each basis vector with respect to its base ones (sum and difference of the frequencies of the input and the output representations)

        Returns:
            the basis built

        """
    
        if self.fibergroup.order() > 0:
            maximum_frequency = None
            maximum_offset = None
    
            if 'maximum_frequency' in kwargs and kwargs['maximum_frequency'] is not None:
                maximum_frequency = kwargs['maximum_frequency']
                assert isinstance(maximum_frequency, int) and maximum_frequency >= 0
    
            if 'maximum_offset' in kwargs and kwargs['maximum_offset'] is not None:
                maximum_offset = kwargs['maximum_offset']
                assert isinstance(maximum_offset, int) and maximum_offset >= 0
    
            assert (maximum_frequency is not None or maximum_offset is not None), \
                'Error! Either the maximum frequency or the maximum offset for the frequencies must be set'
            
            return kernels_CN_act_R2(in_repr, out_repr, rings, sigma,
                                             maximum_frequency,
                                             max_offset=maximum_offset)
        else:
            raise NotImplementedError
            # return kernels.kernels_SO2_act_R2(in_repr, out_repr, rings, sigma)

    def __eq__(self, other):
        if isinstance(other, Rot2dOnR2):
            return self.fibergroup == other.fibergroup
        else:
            return False

    def __hash__(self):
        return hash(self.name)
