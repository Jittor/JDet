from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Iterable, List, Any, Dict
from .math_utils import psi
import numpy as np
import jdet.utils.equivalent as e2nn

class Group(ABC):
    def __init__(self, name: str, continuous: bool, abelian: bool):
        r"""
        Abstract class defining the interface of a group.

        Args:
            name (str): name identifying the group
            continuous (bool): whether the group is non-finite or finite
            abelian (bool): whether the group is *abelian* (commutative)
            
        Attributes:
            ~.name (str): Name identifying the group
            ~.continuous (bool): Whether it is a non-finite or a finite group
            ~.abelian (bool): Whether it is an *abelian* group (i.e. if the group law is commutative)
            ~.identity : Identity element of the group. The identity element :math:`e` satisfies the
                following property :math:`\forall\ g \in G,\ g \cdot e = e \cdot g= g`

        """
        self.name = name
        self.continuous = continuous
        self.abelian = abelian
        self._irreps = {}
        self._representations = {}
        if self.continuous:
            self.elements = None
            self.elements_names = None
        else:
            self.elements = []
            self.elements_names = []
        self.identity = None
        self._subgroups = {}

    def order(self) -> int:
        r"""
        Returns the number of elements in this group if it is a finite group, otherwise -1 is returned        
        """
        if self.elements is not None:
            return len(self.elements)
        else:
            return -1

    @property
    def representations(self) -> Dict[str, e2nn.representation.Representation]:
        r"""
        Dictionary containing all representations (:class:`~e2cnn.group.Representation`)
        instantiated for this group.
        """
        return self._representations
    @property
    def irreps(self) -> Dict[str, e2nn.representation.IrreducibleRepresentation]:
        r"""
        Dictionary containing all irreducible representations (:class:`~e2cnn.group.IrreducibleRepresentation`)
        instantiated for this group.
        """
        return self._irreps
    @property
    def regular_representation(self) -> e2nn.representation.Representation:
        r"""
        Builds the regular representation of the group if the group has a *finite* number of elements;
        returns ``None`` otherwise.
        
        The regular representation of a finite group :math:`G` acts on a vector space :math:`\R^{|G|}` by permuting its
        axes.
        Specifically, associating each axis :math:`e_g` of :math:`\R^{|G|}` to an element :math:`g \in G`, the
        representation of an element :math:`\tilde{g}\in G` is a permutation matrix which maps :math:`e_g` to
        :math:`e_{\tilde{g}g}`.
        For instance, the regular representation of the group :math:`C_4` with elements
        :math:`\{r^k | k=0,\dots,3 \}` is instantiated by:
        
        +-----------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
        |    :math:`g`                      |          :math:`e`                                                                                         |          :math:`r`                                                                                         |        :math:`r^2`                                                                                         |        :math:`r^3`                                                                                         |
        +===================================+============================================================================================================+============================================================================================================+============================================================================================================+============================================================================================================+
        |  :math:`\rho_\text{reg}^{C_4}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\  0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\  0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\  0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ \end{bmatrix}` |
        +-----------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
        
        A vector :math:`v=\sum_g v_g e_g` in :math:`\R^{|G|}` can be interpreted as a scalar function
        :math:`v:G \to \R,\, g \mapsto v_g` on :math:`G`.
        
        Returns:
            the regular representation of the group

        """
        if self.order() < 0:
            raise ValueError(f"Regular representation is supported only for finite groups but "
                             f"the group {self.name} has an infinite number of elements")
        else:
            if "regular" not in self.representations:
                irreps, change_of_basis, change_of_basis_inv = e2nn.representation.build_regular_representation(self)
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'concatenated']
                self.representations["regular"] = e2nn.representation.Representation(self,
                                                                "regular",
                                                                [r.name for r in irreps],
                                                                change_of_basis,
                                                                supported_nonlinearities,
                                                                change_of_basis_inv=change_of_basis_inv,
                                                                )
            return self.representations["regular"]

    @abstractmethod
    def combine(self, e1, e2):
        r"""

        Method that returns the combination of two group elements according to the *group law*.
        
        Args:
            e1: an element of the group
            e2: another element of the group
    
        Returns:
            the group element :math:`e_1 \cdot e_2`
            
        """
        pass

    @abstractmethod
    def inverse(self, element):
        r"""
        Method that returns the inverse in the group of the element given as input

        Args:
            element: an element of the group

        Returns:
            its inverse
        """
        pass
    @property
    @abstractmethod
    def trivial_representation(self) -> e2nn.representation.IrreducibleRepresentation:
        r"""
        Builds the trivial representation of the group.
        The trivial representation is a 1-dimensional representation which maps any element to 1,
        i.e. :math:`\forall g \in G,\ \rho(g) = 1`.
        
        Returns:
            the trivial representation of the group

        """
        pass

    @abstractmethod
    def is_element(self, element) -> bool:
        r"""
        Check whether the input is an element of this group or not.

        Args:
            element: input object to test

        Returns:
            if the input is an element of the group

        """
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

