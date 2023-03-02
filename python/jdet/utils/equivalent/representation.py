from __future__ import annotations
from .group import Group
from typing import Callable, Any, List, Tuple, Dict, Union, Set
import numpy as np
import scipy as sp
from scipy import linalg
import math

__all__ = ['Representation', 'IrreducibleRepresentation', 'build_regular_representation']

class Representation:
    
    def __init__(self,
                 group: Group,
                 name: str,
                 irreps: List[str],
                 change_of_basis: np.ndarray,
                 supported_nonlinearities: Union[List[str], Set[str]],
                 representation: Union[Dict[Any, np.ndarray], Callable[[Any], np.ndarray]] = None,
                 character: Union[Dict[Any, float], Callable[[Any], float]] = None,
                 change_of_basis_inv: np.ndarray = None,
                 **kwargs):
        r"""
        Class used to describe a group representation.
        
        A (real) representation :math:`\rho` of a group :math:`G` on a vector space :math:`V=\mathbb{R}^n` is a map
        (a *homomorphism*) from the group elements to invertible matrices of shape :math:`n \times n`, i.e.:
        
        .. math::
            \rho : G \to \GL{V}
            
        such that the group composition is modeled by a matrix multiplication:
        
        .. math::
            \rho(g_1 g_2) = \rho(g_1) \rho(g_2) \qquad  \forall \ g_1, g_2 \in G \ .
        
        Any representation (of a compact group) can be decomposed into the *direct sum* of smaller, irreducible
        representations (*irreps*) of the group up to a change of basis:
        
        .. math::
            \forall \ g \in G, \ \rho(g) = Q \left( \bigoplus\nolimits_{i \in I} \psi_i(g) \right) Q^{-1} \ .
        
        Here :math:`I` is an index set over the irreps of the group :math:`G` which are contained in the
        representation :math:`\rho`.
        
        This property enables one to study a representation by its irreps and it is used here to work with arbitrary
        representations.
        
        :attr:`e2cnn.group.Representation.change_of_basis` contains the change of basis matrix :math:`Q` while
        :attr:`e2cnn.group.Representation.irreps` is an ordered list containing the names of the irreps :math:`\psi_i`
        indexed by the index set :math:`I`.
        
        A ``Representation`` instance can be used to describe a feature field in a feature map.
        It is the building block to build the representation of a feature map, by "stacking" multiple representations
        (taking their *direct sum*).
        
        .. note ::
            In most of the cases, it should not be necessary to manually instantiate this class.
            Indeed, the user can build the most common representations or some custom representations via the following
            methods and functions:
            
            - :meth:`e2cnn.group.Group.irrep`,
            
            - :meth:`e2cnn.group.Group.regular_representation`,
            
            - :meth:`e2cnn.group.Group.quotient_representation`,
            
            - :meth:`e2cnn.group.Group.induced_representation`,
            
            - :meth:`e2cnn.group.Group.restrict_representation`,
            
            - :func:`e2cnn.group.directsum`,
            
            - :func:`e2cnn.group.change_basis`
            
        If ``representation`` is ``None`` (default), it is automatically inferred by evaluating each irrep, stacking
        their results (through direct sum) and then applying the changes of basis. Warning: the representation of an
        element is built at run-time every time this object is called (through ``__call__``) and this approach might
        become computationally expensive with large representations.
        
        Analogously, if the ``character`` of the representation is ``None`` (default), it is automatically inferred
        evaluating ``representation`` and computing its trace.
        
        .. todo::
            improve the interface for "supported non-linearities" and write somewhere the available options
        
        Args:
            group (Group): the group to be represented.
            name (str): an identification name for this representation.
            irreps (list): a list of strings. Each string represents the name of one of the *irreps* of the
                    group (see :attr:`e2cnn.group.Group.irreps`).
            change_of_basis (~numpy.ndarray): the matrix which transforms the direct sum of the irreps
                    in this representation.
            supported_nonlinearities (list or set): a list or set of nonlinearity types supported by this
                    representation.
            representation (dict or callable, optional): a callable implementing this representation or a dictionary
                    mapping each of the group's elements to its representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dictionary mapping each element to its character.
            change_of_basis_inv (~numpy.ndarray, optional): the inverse of the ``change_of_basis`` matrix; if not
                    provided (``None``), it is computed from ``change_of_basis``.
            **kwargs: custom attributes the user can set and, then, access from the dictionary in
                    :attr:`e2cnn.group.Representation.attributes`
            
        Attributes:
            ~.group (Group): The group which is being represented.
            ~.name (str): A string identifying this representation.
            ~.size (int): Dimensionality of the vector space of this representation. In practice, this is the size of the
                matrices this representation maps the group elements to.
            ~.change_of_basis (~numpy.ndarray): Change of basis matrix for the irreps decomposition.
            ~.change_of_basis_inv (~numpy.ndarray): Inverse of the change of basis matrix for the irreps decomposition.
            ~.representation (callable): Method implementing the map from group elements to their representation matrix.
            ~.supported_nonlinearities (set): A set of strings identifying the non linearities types supported by this representation.
            ~.irreps (list): List of irreps into which this representation decomposes.
            ~.attributes (dict): Custom attributes set when creating the instance of this class.
            ~.irreducible (bool): Whether this is an irreducible representation or not (i.e. if it can't be decomposed into further invariant subspaces).        
        """
        
        assert len(change_of_basis.shape) == 2 and change_of_basis.shape[0] == change_of_basis.shape[1]
        # can't have the name of an already existing representation
        assert name not in group.representations, f"A representation for {group.name} with name {name} already exists!"
        if change_of_basis_inv is None:
            change_of_basis_inv = sp.linalg.inv(change_of_basis)
        assert len(change_of_basis_inv.shape) == 2
        assert change_of_basis_inv.shape[0] == change_of_basis.shape[0]
        assert change_of_basis_inv.shape[1] == change_of_basis.shape[1]
        assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(change_of_basis.shape[0]))
        assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(change_of_basis.shape[0]))
        
        # Group: A string identifying this representation.
        self.group = group
        
        # str: The group this is a representation of.
        self.name = name
        
        # int: Dimensionality of the vector space of this representation.
        # In practice, this is the size of the matrices this representation maps the group elements to.
        self.size = change_of_basis.shape[0]
        
        # np.ndarray: Change of basis matrix for the irreps decomposition.
        self.change_of_basis = change_of_basis

        # np.ndarray: Inverse of the change of basis matrix for the irreps decomposition.
        self.change_of_basis_inv = change_of_basis_inv

        if representation is None:
            irreps_instances = [group.irreps[n] for n in irreps]
            representation = direct_sum_factory(irreps_instances, change_of_basis, change_of_basis_inv)
        elif isinstance(representation, dict):
            assert set(representation.keys()) == set(self.group.elements), "Error! Keys don't match group's elements"
            
            self._stored_representations = representation
            representation = lambda e, repr=self: repr._stored_representations[e]
            
        elif not callable(representation):
            raise ValueError('Error! "representation" is neither a dictionary nor callable')
        
        # Callable: Method implementing the map from group elements to matrix representations.
        self.representation = representation

        if isinstance(character, dict):
            
            assert set(character.keys()) == set(self.group.elements), "Error! Keys don't match group's elements"
            
            self._characters = character

        elif callable(character):
            self._characters = character
        elif character is None:
            # if the character is not given as input, it is automatically inferred from the given representation
            # taking its trace
            self._characters = None
        else:
            raise ValueError('Error! "character" must be a dictionary, a callable or "None"')

        # TODO - assert size matches size of the matrix returned by the callable
        
        # list(str): List of irreps this representation decomposes into
        self.irreps = irreps
        self.supported_nonlinearities = set(supported_nonlinearities)
        
        # dict: Custom attributes set when creating the instance of this class
        self.attributes = kwargs

        # TODO : remove the condition of an identity change of basis?
        # bool: Whether this is an irreducible representation or not (i.e.: if it can't be decomposed further)
        self.irreducible = len(self.irreps) == 1 and np.allclose(self.change_of_basis, np.eye(self.change_of_basis.shape[0]))

    def character(self, e) -> float:
        r"""

        The *character* of a finite-dimensional real representation is a function mapping a group element
        to the trace of its representation:
        .. math::
            \chi_\rho: G \to \mathbb{C}, \ \ g \mapsto \chi_\rho(g) := \operatorname{tr}(\rho(g))

        It is useful to perform the irreps decomposition of a representation using *Character Theory*.
        Args:
            e: an element of the group of this representation
        Returns:
            the character of the element
        """
        
        if self._characters is None:
            # if the character is not given as input, it is automatically inferred from the given representation
            # taking its trace
            repr = self(e)
            return np.trace(repr)
        elif isinstance(self._characters, dict):
            return self._characters[e]

        elif callable(self._characters):
            return self._characters(e)
        else:
            raise RuntimeError('Error! Character not recognized!')
    def is_trivial(self) -> bool:
        r"""
        Whether this representation is trivial or not.
        """
        return self.irreducible and self.group.trivial_representation.name == self.irreps[0]
    def __call__(self, element) -> np.ndarray:
        """
        An instance of this class can be called and it implements the mapping from an element of a group to its
        representation.
        
        This is equivalent to calling :meth:`e2cnn.group.Representation.representation`,
        though ``__call__`` first checks ``element`` is a valid input (i.e. an element of the group).
        It is recommended to use this call.

        Args:
            element: an element of the group

        Returns:
            A matrix representing the input element

        """
        
        assert self.group.is_element(element), f"{self.group.name}, {element}: {self.group.is_element(element)}"
        return self.representation(element)

    def __eq__(self, other: Representation) -> bool:
        if not isinstance(other, Representation):
            return False
        
        return (self.name == other.name
                and self.group == other.group
                and np.allclose(self.change_of_basis, other.change_of_basis)
                and self.irreps == other.irreps
                and self.supported_nonlinearities == other.supported_nonlinearities)
    
    def __repr__(self) -> str:
        return f"{self.group.name}|{self.name}:{self.size},{len(self.irreps)},{self.change_of_basis.sum()}"
    
    def __hash__(self):
        return hash(repr(self))


class IrreducibleRepresentation(Representation):
    def __init__(self,
                 group: Group,
                 name: str,
                 representation: Union[Dict[Any, np.ndarray], Callable[[Any], np.ndarray]],
                 size: int,
                 sum_of_squares_constituents: int,
                 supported_nonlinearities: List[str],
                 character: Union[Dict[Any, float], Callable[[Any], float]] = None,
                 **kwargs
                 ):
        """
        Describes an "*irreducible representation*" (*irrep*).
        Irreducible representations are the building blocks into which any other representation decomposes under a
        change of basis.
        Indeed, any :class:`~e2cnn.group.Representation` is internally decomposed into a direct sum of irreps.
        
        Args:
            group (Group): the group which is being represented
            name (str): an identification name for this representation
            representation (dict or callable): a callable implementing this representation or a dictionary
                    mapping each of the group's elements to its representation.
            size (int): the size of the vector space where this representation is defined (i.e. the size of the matrices)
            sum_of_squares_constituents (int): the sum of the squares of the multiplicities of pairwise distinct
                        irreducible constituents of the character of this representation over a non-splitting field
            supported_nonlinearities (list): list of nonlinearitiy types supported by this representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dictionary mapping each element to its character.
            **kwargs: custom attributes the user can set and, then, access from the dictionary
                    in :attr:`e2cnn.group.Representation.attributes`
        
        Attributes:
            sum_of_squares_constituents (int): the sum of the squares of the multiplicities of pairwise distinct
                    irreducible constituents of the character of this representation over a non-splitting field (see
                    `Character Orthogonality Theorem <https://groupprops.subwiki.org/wiki/Character_orthogonality_theorem#Statement_over_general_fields_in_terms_of_inner_product_of_class_functions>`_
                    over general fields)
            
        """
        
        super(IrreducibleRepresentation, self).__init__(group,
                                                        name,
                                                        [name],
                                                        np.eye(size),
                                                        supported_nonlinearities,
                                                        representation=representation,
                                                        character=character,
                                                        **kwargs)
        self.irreducible = True
        self.sum_of_squares_constituents = sum_of_squares_constituents

def direct_sum_factory(irreps: List[IrreducibleRepresentation],
                       change_of_basis: np.ndarray,
                       change_of_basis_inv: np.ndarray = None
                       ) -> Callable[[Any], np.ndarray]:
    """
    The method builds and returns a function implementing the direct sum of the "irreps" transformed by the given
    "change_of_basis" matrix.

    More precisely, the built method will take as input a value accepted by all the irreps, evaluate the irreps on that
    input and return the direct sum of the produced matrices left and right multiplied respectively by the
    change_of_basis matrix and its inverse.

    Args:
        irreps (list): list of irreps
        change_of_basis: the matrix transforming the direct sum of the irreps
        change_of_basis_inv: the inverse of the change of basis matrix

    Returns:
        function taking an input accepted by the irreps and returning the direct sum of the irreps evaluated
        on that input
    """
    
    shape = change_of_basis.shape
    assert len(shape) == 2 and shape[0] == shape[1]
    
    if change_of_basis_inv is None:
        # pre-compute the inverse of the change-of-_bases matrix
        change_of_basis_inv = np.linalg.inv(change_of_basis)
    else:
        assert len(change_of_basis_inv.shape) == 2
        assert change_of_basis_inv.shape[0] == change_of_basis.shape[0]
        assert change_of_basis_inv.shape[1] == change_of_basis.shape[1]
        assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(change_of_basis.shape[0]))
        assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(change_of_basis.shape[0]))
    
    unique_irreps = list({irr.name: irr for irr in irreps}.items())
    irreps_names = [irr.name for irr in irreps]
    
    def direct_sum(element,
                   irreps_names=irreps_names, change_of_basis=change_of_basis,
                   change_of_basis_inv=change_of_basis_inv, unique_irreps=unique_irreps):
        reprs = {}
        for n, irr in unique_irreps:
            reprs[n] = irr(element)
        
        blocks = []
        for irrep_name in irreps_names:
            repr = reprs[irrep_name]
            blocks.append(repr)
        
        P = sp.sparse.block_diag(blocks, format='csc')
        
        return change_of_basis @ P @ change_of_basis_inv
    
    return direct_sum

def directsum(reprs: List[Representation],
              change_of_basis: np.ndarray = None,
              name: str = None
              ) -> Representation:
    r"""

    Compute the *direct sum* of a list of representations of a group.
    
    The direct sum of two representations is defined as follow:
    
    .. math::
        \rho_1(g) \oplus \rho_2(g) = \begin{bmatrix} \rho_1(g) & 0 \\ 0 & \rho_2(g) \end{bmatrix}
    
    This can be generalized to multiple representations as:
    
    .. math::
        \bigoplus_{i=1}^I \rho_i(g) = (\rho_1(g) \oplus (\rho_2(g) \oplus (\rho_3(g) \oplus \dots = \begin{bmatrix}
            \rho_1(g) &         0 &  \dots &      0 \\
                    0 & \rho_2(g) &  \dots & \vdots \\
               \vdots &    \vdots & \ddots &      0 \\
                    0 &     \dots &      0 & \rho_I(g) \\
        \end{bmatrix}
    

    .. note::
        All the input representations need to belong to the same group.

    Args:
        reprs (list): the list of representations to sum.
        change_of_basis (~numpy.ndarray, optional): an invertible square matrix to use as change of basis after computing the direct sum.
                By default (``None``), an identity matrix is used, such that only the direct sum is evaluated.
        name (str, optional): a name for the new representation.

    Returns:
        the direct sum

    """
    
    group = reprs[0].group
    for r in reprs:
        assert group == r.group
    
    if name is None:
        name = "_".join([f"[{r.name}]" for r in reprs])
    
    irreps = []
    for r in reprs:
        irreps += r.irreps
    
    size = sum([r.size for r in reprs])
    
    cob = np.zeros((size, size))
    cob_inv = np.zeros((size, size))
    p = 0
    for r in reprs:
        cob[p:p + r.size, p:p + r.size] = r.change_of_basis
        cob_inv[p:p + r.size, p:p + r.size] = r.change_of_basis_inv
        p += r.size

    if change_of_basis is not None:
        change_of_basis = change_of_basis @ cob
        change_of_basis_inv = sp.linalg.inv(change_of_basis)
    else:
        change_of_basis = cob
        change_of_basis_inv = cob_inv
    supported_nonlinearities = set.intersection(*[r.supported_nonlinearities for r in reprs])
    return Representation(group, name, irreps, change_of_basis, supported_nonlinearities, change_of_basis_inv=change_of_basis_inv)


def build_regular_representation(group: Group) -> Tuple[List[IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""
    
    Build the regular representation of the input ``group``.
    As the regular representation has size equal to the number of elements in the group, only
    finite groups are accepted.
    
    Args:
        group (Group): the group whose representations has to be built

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the regular representation

    """
    assert group.order() > 0
    assert group.elements is not None and len(group.elements) > 0
    
    size = group.order()
    index = {e: i for i, e in enumerate(group.elements)}
    representation = {}
    character = {}
    
    for e in group.elements:
        r = np.zeros((size, size), dtype=float)
        for g in group.elements:
            eg = group.combine(e, g)
            i = index[g]
            j = index[eg]
            r[j, i] = 1.0
        
        representation[e] = r
        # the character maps an element to the trace of its representation
        character[e] = np.trace(r)

    # compute the multiplicities of the irreps from the dot product between
    # their characters and the character of the representation
    irreps = []
    multiplicities = []
    for irrep_name, irrep in group.irreps.items():
        # for each irrep
        multiplicity = 0.0
    
        # compute the inner product with the representation's character
        for element, char in character.items():
            multiplicity += char * irrep.character(group.inverse(element))
    
        multiplicity /= len(character) * irrep.sum_of_squares_constituents
    
        # the result has to be an integer
        assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
            "Multiplicity of irrep %s is not an integer: %f" % (irrep_name, multiplicity)
        # print(irrep_name, multiplicity)

        multiplicity = int(round(multiplicity))
        irreps += [irrep]*multiplicity
        multiplicities += [(irrep, multiplicity)]
    P = directsum(irreps, name="irreps")
    v = np.zeros((size, 1), dtype=float)
    p = 0
    for irr, m in multiplicities:
        assert irr.size >= m
        s = irr.size
        v[p:p+m*s, 0] = np.eye(m, s).reshape(-1) * np.sqrt(s)
        p += m*s
        
    change_of_basis = np.zeros((size, size))
    np.set_printoptions(precision=4, threshold=10*size**2, suppress=False, linewidth=25*size + 5)
    for e in group.elements:
        ev = P(e) @ v
        change_of_basis[index[e], :] = ev.T
    change_of_basis /= np.sqrt(size)
    
    # the computed change of basis is an orthonormal matrix
    change_of_basis_inv = change_of_basis.T
    
    return irreps, change_of_basis, change_of_basis_inv
