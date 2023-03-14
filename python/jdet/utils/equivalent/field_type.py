from __future__ import annotations
from .gspace import GSpace
from .group import Group
from .representation import Representation
from typing import List

class FieldType:
    def __init__(self,
                 gspace: GSpace,
                 representations: List[Representation]):
        r"""
        
        An ``FieldType`` can be interpreted as the *data type* of a feature space. It describes:
        
        - the base space on which a feature field is living and its symmetries considered
        - the transformation law of feature fields under the action of the fiber group
        
        The former is formalize by a choice of ``gspace`` while the latter is determined by a choice of group
        representations (``representations``), passed as a list of :class:`~e2cnn.group.Representation` instances.
        Each single representation in this list corresponds to one independent feature field contained in the feature
        space.
        The input ``representations`` need to belong to ``gspace``'s fiber group
        (:attr:`e2cnn.gspaces.GSpace.fibergroup`).
        
        .. note ::
            
            Mathematically, this class describes a *(trivial) vector bundle*, *associated* to the symmetry group
            :math:`(\R^D, +) \rtimes G`.
            
            Given a *principal bundle* :math:`\pi: (\R^D, +) \rtimes G \to \R^D, tg \mapsto tG`
            with fiber group :math:`G`, an *associated vector bundle* has the same base space
            :math:`\R^D` but its fibers are vector spaces like :math:`\mathbb{R}^c`.
            Moreover, these vector spaces are associated to a :math:`c`-dimensional representation :math:`\rho` of the
            fiber group :math:`G` and transform accordingly.
            
            The representation :math:`\rho` is defined as the *direct sum* of the representations :math:`\{\rho_i\}_i`
            in ``representations``. See also :func:`~e2cnn.group.directsum`.
            
        
        Args:
            gspace (GSpace): the space where the feature fields live and its symmetries
            representations (list): a list of :class:`~e2cnn.group.Representation` s of the ``gspace``'s fiber group,
                            determining the transformation laws of the feature fields
        
        Attributes:
            ~.gspace (GSpace)
            ~.representations (list)
            ~.size (int): dimensionality of the feature space described by the :class:`~e2cnn.nn.FieldType`.
                          It corresponds to the sum of the dimensionalities of the individual feature fields or
                          group representations (:attr:`e2cnn.group.Representation.size`).
 
            
        """
        assert len(representations) > 0
        
        for repr in representations:
            assert repr.group == gspace.fibergroup
        
        # GSpace: Space where data lives and its (abstract) symmetries
        self.gspace = gspace
        
        # list: List of representations of each feature field composing the feature space of this type
        self.representations = representations
        
        # int: size of the field associated to this type.
        # as the representation associated to the field is the direct sum of the representations
        # in :attr:`e2cnn.nn.fieldtype.representations`, its size is the sum of each of these
        # representations' size
        self.size = sum([repr.size for repr in representations])
        self._unique_representations = set(self.representations)
        self._representation = None
        self._field_start = None
        self._field_end = None
        self._hash = hash(self.gspace.name + ': {' + ', '.join([r.name for r in self.representations]) + '}')

    def index_select(self, index: List[int]) -> 'FieldType':
        r"""
        Build a new :class:`~e2cnn.nn.FieldType` from the current one by taking the
        :class:`~e2cnn.group.Representation` s selected by the input ``index``.
        Args:
            index (list): a list of integers in the range ``{0, ..., N-1}``, where ``N`` is the number of representations
                          in the current field type
        Returns:
            the new field type
        """
        assert max(index) < len(self.representations)
        assert min(index) >= 0

        # retrieve the fields in the input representation to build the output representation
        representations = [self.representations[i] for i in index]
        return FieldType(self.gspace, representations)

    @property
    def fibergroup(self) -> Group:
        r"""
        The fiber group of :attr:`~e2cnn.nn.FieldType.gspace`.

        Returns:
            the fiber group

        """
        return self.gspace.fibergroup

    def __len__(self) -> int:
        r"""
        Return the number of feature fields in this :class:`~e2cnn.nn.FieldType`, i.e. the length of
        :attr:`e2cnn.nn.FieldType.representations`.
        .. note ::
            This is in general different from :attr:`e2cnn.nn.FieldType.size`.
        Returns:
            the number of fields in this type
        """
        return len(self.representations)

    def __iter__(self):
        r"""
        It is possible to iterate over all :attr:`~e2cnn.nn.FieldType.representations` in a field type by using
        :class:`~e2cnn.nn.FieldType` as an *iterable* object.
        """
        return iter(self.representations)

    def __eq__(self, other):
        if isinstance(other, FieldType):
            return self.gspace == other.gspace and self.representations == other.representations
        else:
            return False

    def __hash__(self):
        return self._hash
    
    def __repr__(self):
        return '[' + self.gspace.name + ': {' + ', '.join([r.name for r in self.representations]) + '}]'
