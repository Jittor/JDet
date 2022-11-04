import jittor as jt
from jittor import nn
from jdet.utils.equivalent import FieldType, GeneralOnR2, Representation, check_consecutive_numbers
from jdet.utils.equivalent.kernels.basis import Basis, EmptyBasisException
from typing import Callable, Iterable, List, Union, Dict
from collections import defaultdict
import jittor as jt
import numpy as np

__all__ = ["BlocksBasisExpansion"]

def normalize_basis(basis: jt.Var, sizes: jt.Var) -> jt.Var:
    r"""
    Normalize the filters in the input tensor.
    The tensor of shape :math:`(B, O, I, ...)` is interpreted as a basis containing ``B`` filters/elements, each with
    ``I`` inputs and ``O`` outputs. The spatial dimensions ``...`` can be anything.
    .. notice ::
        Notice that the method changes the input tensor inplace
    Args:
        basis (torch.Tensor): tensor containing the basis to normalize
        sizes (torch.Tensor): original input size of the basis elements, without the padding and the change of basis
    Returns:
        the normalized basis (the operation is done inplace, so this is ust a reference to the input tensor)
    """
    b = basis.shape[0]
    assert len(basis.shape) > 2
    assert sizes.shape == (b,)
    # compute the norm of each basis vector
    assert len(basis.shape) == 4
    # norms = torch.einsum('bop...,bpq...->boq...', (basis, basis.transpose(1, 2)))
    basis = jt.transpose(basis, (3,0,1,2))
    # Warning: possible accuracy loss
    basis = jt.array(basis, dtype=jt.float32)
    norms = jt.matmul(basis, basis.transpose(2, 3))
    basis = jt.transpose(basis, (1,2,3,0))
    norms = jt.transpose(norms, (1,2,3,0))
    
    # Removing the change of basis, these matrices should be multiples of the identity
    # where the scalar on the diagonal is the variance
    # in order to find this variance, we can compute the trace (which is invariant to the change of basis)
    # and divide by the number of elements in the diagonal ignoring the padding.
    # Therefore, we need to know the original size of each basis element.
    # norms = torch.einsum("bii...->b", norms)
    norms = jt.reindex(norms, [norms.shape[0], norms.shape[1], norms.shape[3]], ['i0', 'i1', 'i1', 'i2'])
    norms = jt.sum(norms, dims=(1,2), keepdims=False)
    norms /= sizes
    norms[norms < 1e-15] = 0
    norms = jt.sqrt(norms)
    norms[norms < 1e-6] = 1
    norms[norms != norms] = 1
    norms = norms.view(b, *([1] * (len(basis.shape) - 1)))
    
    # divide by the norm
    basis /= norms

    return basis

class SingleBlockBasisExpansion(nn.Module):    
    def __init__(self,
                 basis: Basis,
                 points: np.ndarray,
                 basis_filter: Callable[[dict], bool] = None,
                 ):
        r"""
        Basis expansion method for a single contiguous block, i.e. for kernels/PDOs whose input type and output type contain
        only fields of one type.
        This class should be instantiated through the factory method
        :func:`~e2cnn.nn.modules.r2_conv.block_basisexpansion` to enable caching.
        Args:
            basis (Basis): analytical basis to sample
            points (ndarray): points where the analytical basis should be sampled
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            
        """
        super(SingleBlockBasisExpansion, self).__init__()
        self.basis = basis
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
        if not any(mask):
            raise EmptyBasisException
        attributes = [attr for b, attr in enumerate(basis) if mask[b]]
        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in attributes:
            sizes.append(attr["shape"][0])
        # sample the basis on the grid
        # and filter out the basis elements discarded by the filter
        sampled_basis = jt.array(basis.sample_masked(points, mask=mask)).permute(2, 0, 1, 3)

        # DEPRECATED FROM PyTorch 1.2
        # PyTorch 1.2 suggests using BoolTensor instead of ByteTensor for boolean indexing
        # but BoolTensor have been introduced only in PyTorch 1.2
        # Hence, for the moment we use ByteTensor

        # normalize the basis
        sizes = jt.array(sizes, dtype=sampled_basis.dtype)
        assert sizes.shape[0] == sampled_basis.shape[0], (sizes.shape, sampled_basis.shape)
        sampled_basis = normalize_basis(sampled_basis, sizes)

        # discard the basis which are close to zero everywhere
        norms = (sampled_basis ** 2).reshape(sampled_basis.shape[0], -1).sum(1) > 1e-2
        if not jt.any(norms).item():
            raise EmptyBasisException
        sampled_basis = sampled_basis[norms, ...]
        
        self.attributes = [attr for b, attr in enumerate(attributes) if norms[b]]
        # register the bases tensors as parameters of this module
        self.sampled_basis = sampled_basis
        self.sampled_basis.stop_grad()
        # self.register_buffer('sampled_basis', sampled_basis)
        self._idx_to_ids = []
        self._ids_to_idx = {}
        for idx, attr in enumerate(self.attributes):
            if "radius" in attr:
                radial_info = attr["radius"]
            elif "order" in attr:
                radial_info = attr["order"]
            else:
                raise ValueError("No radial information found.")
            id = '({}-{},{}-{})_({}/{})_{}'.format(
                    attr["in_irrep"], attr["in_irrep_idx"],  # name and index within the field of the input irrep
                    attr["out_irrep"], attr["out_irrep_idx"],  # name and index within the field of the output irrep
                    radial_info,
                    attr["frequency"],  # frequency of the basis element
                    # int(np.abs(attr["frequency"])),  # absolute frequency of the basis element
                    attr["inner_idx"],
                    # index of the basis element within the basis of radially independent kernels between the irreps
                )
            attr["id"] = id
            self._ids_to_idx[id] = idx
            self._idx_to_ids.append(id)

    def execute(self, weights: jt.Var) -> jt.Var:
    
        assert len(weights.shape) == 2 and weights.shape[1] == self.dimension()
    
        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        # return torch.einsum('boi...,kb->koi...', self.sampled_basis, weights) #.transpose(1, 2).contiguous()
        assert len(self.sampled_basis.shape) == 4
        sampled = jt.transpose(self.sampled_basis, (1,2,3,0))
        result = jt.nn.matmul_transpose(sampled, weights)
        result = jt.transpose(result, (3,0,1,2))
        return result

    def get_basis_names(self) -> List[str]:
        return self._idx_to_ids

    def get_element_info(self, name: Union[str, int]) -> Dict:
        if isinstance(name, str):
            name = self._ids_to_idx[name]
        return self.attributes[name]

    def get_basis_info(self) -> Iterable:
        return iter(self.attributes)

    def dimension(self) -> int:
        return self.sampled_basis.shape[0]

    # def __eq__(self, other):
    #     if isinstance(other, SingleBlockBasisExpansion):
    #         return (
    #                 self.basis == other.basis and
    #                 jt.allclose(self.sampled_basis, other.sampled_basis) and
    #                 (self._mask == other._mask).all()
    #         )
    #     else:
    #         return False

    # def __hash__(self):
    #     return 10000 * hash(self.basis) + 100 * hash(self.sampled_basis) + hash(self._mask)



_stored_filters = {}

def block_basisexpansion(basis: Basis,
                         points: np.ndarray,
                         basis_filter: Callable[[dict], bool] = None,
                         recompute: bool = False
                         ) -> SingleBlockBasisExpansion:
    r"""
    Return an instance of :class:`~e2cnn.nn.modules.r2_conv.SingleBlockBasisExpansion`.
    This function support caching through the argument ``recompute``.
    Args:
        basis (Basis): basis defining the space of kernels
        points (~np.ndarray): points where the analytical basis should be sampled
        basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                           element's attributes and return whether to keep it or not.
        recompute (bool, optional): whether to recompute new bases (``True``) or reuse, if possible,
                                    already built tensors (``False``, default).
    """
    
    if not recompute:
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
        key = (basis, mask.tobytes(), points.tobytes())
        if key not in _stored_filters:
            _stored_filters[key] = SingleBlockBasisExpansion(basis, points, basis_filter)
        return _stored_filters[key]
    
    else:
        return SingleBlockBasisExpansion(basis, points, basis_filter)

def _retrieve_indices(type: FieldType):
    fiber_position = 0
    _indices = defaultdict(list)
    _count = defaultdict(int)
    _contiguous = {}
    for repr in type.representations:
        _indices[repr.name] += list(range(fiber_position, fiber_position + repr.size))
        fiber_position += repr.size
        _count[repr.name] += 1
    for name, indices in _indices.items():
        # _contiguous[o_name] = indices == list(range(indices[0], indices[0]+len(indices)))
        _contiguous[name] = check_consecutive_numbers(indices)
        _indices[name] = jt.array(indices, dtype=jt.int64)
    return _count, _indices, _contiguous

def _compute_attrs_and_ids(in_type, out_type, block_submodules):
    basis_ids = defaultdict(lambda: [])
    # iterate over all blocks
    # each block is associated to an input/output representations pair
    out_fiber_position = 0
    out_irreps_count = 0
    for o, o_repr in enumerate(out_type.representations):
        in_fiber_position = 0
        in_irreps_count = 0
        for i, i_repr in enumerate(in_type.representations):
            reprs_names = (i_repr.name, o_repr.name)
            # if a basis for the space of kernels between the current pair of representations exists
            if reprs_names in block_submodules:
                # retrieve the attributes of each basis element and build a new list of
                # attributes adding information specific to the current block
                ids = []
                for attr in block_submodules[reprs_names].get_basis_info():
                    # build the ids of the basis vectors
                    # add names and indices of the input and output fields
                    id = '({}-{},{}-{})'.format(i_repr.name, i, o_repr.name, o)
                    # add the original id in the block submodule
                    id += "_" + attr["id"]
                    ids.append(id)
                # append the ids of the basis vectors
                basis_ids[reprs_names] += ids
            in_fiber_position += i_repr.size
            in_irreps_count += len(i_repr.irreps)
        out_fiber_position += o_repr.size
        out_irreps_count += len(o_repr.irreps)
    return basis_ids

class BlocksBasisExpansion(nn.Module):
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 basis_generator: Callable[[Representation, Representation], Basis],
                 points: np.ndarray,
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 **kwargs
                 ):
        r"""
        With this algorithm, the expansion is done on the intertwiners of the fields' representations pairs in input and
        output.
        Args:
            in_type (FieldType): the input field type
            out_type (FieldType): the output field type
            basis_generator (callable): method that generates the analytical filter basis
            points (~numpy.ndarray): points where the analytical basis should be sampled
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.
            **kwargs: keyword arguments to be passed to ```basis_generator```
        Attributes:
            S (int): number of points where the filters are sampled
        """

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)
        super(BlocksBasisExpansion, self).__init__()
        self._in_type = in_type
        self._out_type = out_type
        self._input_size = in_type.size
        self._output_size = out_type.size
        self.points = points
        # int: number of points where the filters are sampled
        self.S = self.points.shape[1]
        # we group the basis vectors by their input and output representations
        _block_expansion_modules = {}
        # iterate through all different pairs of input/output representationions
        # and, for each of them, build a basis
        for i_repr in in_type._unique_representations:
            for o_repr in out_type._unique_representations:
                reprs_names = (i_repr.name, o_repr.name)
                try:
                    basis = basis_generator(i_repr, o_repr, **kwargs)
                    block_expansion = block_basisexpansion(basis, points, basis_filter, recompute=recompute)
                    _block_expansion_modules[reprs_names] = block_expansion
                    self.__setattr__(f"block_expansion_{reprs_names}", block_expansion)
                except EmptyBasisException:
                    pass
        if len(_block_expansion_modules) == 0:
            print('WARNING! The basis for the block expansion of the filter is empty!')
        self._n_pairs = len(in_type._unique_representations) * len(out_type._unique_representations)

        # the list of all pairs of input/output representations which don't have an empty basis
        self._representations_pairs = sorted(list(_block_expansion_modules.keys()))
        # retrieve for each representation in both input and output fields:
        # - the number of its occurrences,
        # - the indices where it occurs and
        # - whether its occurrences are contiguous or not
        self._in_count, _in_indices, _in_contiguous = _retrieve_indices(in_type)
        self._out_count, _out_indices, _out_contiguous = _retrieve_indices(out_type)
        
        # compute the attributes and an id for each basis element (and, so, of each parameter)
        basis_ids = _compute_attrs_and_ids(in_type, out_type, _block_expansion_modules)
        
        self._weights_ranges = {}
        last_weight_position = 0
        self._ids_to_basis = {}
        self._basis_to_ids = []
        self._contiguous = {}
        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        for io_pair in self._representations_pairs:
            self._contiguous[io_pair] = _in_contiguous[io_pair[0]] and _out_contiguous[io_pair[1]]
            # build the indices tensors
            if self._contiguous[io_pair]:
                in_indices = [
                    _in_indices[io_pair[0]].min(),
                    _in_indices[io_pair[0]].max() + 1,
                    _in_indices[io_pair[0]].max() + 1 - _in_indices[io_pair[0]].min()
                ]
                out_indices = [
                    _out_indices[io_pair[1]].min(),
                    _out_indices[io_pair[1]].max() + 1,
                    _out_indices[io_pair[1]].max() + 1 - _out_indices[io_pair[1]].min()
                ]
                
                setattr(self, 'in_indices_{}'.format(io_pair), in_indices)
                setattr(self, 'out_indices_{}'.format(io_pair), out_indices)

            else:
                out_indices, in_indices = jt.meshgrid([_out_indices[io_pair[1]], _in_indices[io_pair[0]]])
                in_indices = in_indices.reshape(-1)
                out_indices = out_indices.reshape(-1)
                # register the indices tensors and the bases tensors as parameters of this module
                # self.register_buffer('in_indices_{}'.format(io_pair), in_indices)
                # self.register_buffer('out_indices_{}'.format(io_pair), out_indices)
                in_indices.stop_grad()
                out_indices.stop_grad()
                self.__setattr__('in_indices_{}'.format(io_pair), in_indices)
                self.__setattr__('out_indices_{}'.format(io_pair), out_indices)
            # count the actual number of parameters
            total_weights = len(basis_ids[io_pair])
            for i, id in enumerate(basis_ids[io_pair]):
                self._ids_to_basis[id] = last_weight_position + i
            self._basis_to_ids += basis_ids[io_pair]
            # evaluate the indices in the global weights tensor to use for the basis belonging to this group
            self._weights_ranges[io_pair] = (last_weight_position, last_weight_position + total_weights)
            # increment the position counter
            last_weight_position += total_weights

    def dimension(self) -> int:
        return len(self._ids_to_basis)

    def get_basis_info(self) -> Iterable:
        out_irreps_counts = [0]
        out_block_counts = defaultdict(list)
        for o, o_repr in enumerate(self._out_type.representations):
            out_irreps_counts.append(out_irreps_counts[-1] + len(o_repr.irreps))
            out_block_counts[o_repr.name].append(o)
            
        in_irreps_counts = [0]
        in_block_counts = defaultdict(list)
        for i, i_repr in enumerate(self._in_type.representations):
            in_irreps_counts.append(in_irreps_counts[-1] + len(i_repr.irreps))
            in_block_counts[i_repr.name].append(i)

        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        idx = 0
        for reprs_names in self._representations_pairs:
            block_expansion = getattr(self, f"block_expansion_{reprs_names}")
            for o in out_block_counts[reprs_names[1]]:
                out_irreps_count = out_irreps_counts[o]
                for i in in_block_counts[reprs_names[0]]:
                    in_irreps_count = in_irreps_counts[i]
                    # retrieve the attributes of each basis element and build a new list of
                    # attributes adding information specific to the current block
                    for attr in block_expansion.get_basis_info():
                        attr = attr.copy()
                        attr.update({
                            "in_irreps_position": in_irreps_count + attr["in_irrep_idx"],
                            "out_irreps_position": out_irreps_count + attr["out_irrep_idx"],
                            "in_repr": reprs_names[0],
                            "out_repr": reprs_names[1],
                            "in_field_position": i,
                            "out_field_position": o,
                        })
                        # build the ids of the basis vectors
                        # add names and indices of the input and output fields
                        id = '({}-{},{}-{})'.format(reprs_names[0], i, reprs_names[1], o)
                        # add the original id in the block submodule
                        id += "_" + attr["id"]
                
                        # update with the new id
                        attr["id"] = id
                        
                        attr["idx"] = idx
                        idx += 1
                
                        yield attr

    def _expand_block(self, weights, io_pair):
        # retrieve the basis
        block_expansion = getattr(self, f"block_expansion_{io_pair}")

        # retrieve the linear coefficients for the basis expansion
        coefficients = weights[self._weights_ranges[io_pair][0]:self._weights_ranges[io_pair][1]]
    
        # reshape coefficients for the batch matrix multiplication
        coefficients = coefficients.view(-1, block_expansion.dimension())
        
        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        _filter = block_expansion(coefficients)
        k, o, i, p = _filter.shape
        
        _filter = _filter.view(
            self._out_count[io_pair[1]],
            self._in_count[io_pair[0]],
            o,
            i,
            self.S,
        )
        _filter = _filter.transpose(1, 2)
        return _filter

    def execute(self, weights: jt.Var) -> jt.Var:
        """
        Forward step of the Module which expands the basis and returns the filter built
        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis filters

        Returns:
            the filter built

        """
        assert weights.shape[0] == self.dimension()
        assert len(weights.shape) == 1
    
        if self._n_pairs == 1:
            # if there is only one block (i.e. one type of input field and one type of output field),
            #  we can return the expanded block immediately, instead of copying it inside a preallocated empty tensor
            io_pair = self._representations_pairs[0]
            in_indices = getattr(self, f"in_indices_{io_pair}")
            out_indices = getattr(self, f"out_indices_{io_pair}")
            _filter = self._expand_block(weights, io_pair)
            _filter = jt.reshape(_filter, (out_indices[2].item(), in_indices[2].item(), self.S))
        else:
            # build the tensor which will contain te filter
            _filter = jt.zeros(self._output_size, self._input_size, self.S, device=weights.device)

            # iterate through all input-output field representations pairs
            for io_pair in self._representations_pairs:
                
                # retrieve the indices
                in_indices = getattr(self, f"in_indices_{io_pair}")
                out_indices = getattr(self, f"out_indices_{io_pair}")
                
                # expand the current subset of basis vectors and set the result in the appropriate place in the filter
                expanded = self._expand_block(weights, io_pair)
                
                if self._contiguous[io_pair]:
                    _filter[
                        out_indices[0]:out_indices[1],
                        in_indices[0]:in_indices[1],
                        :,
                    ] = expanded.reshape(out_indices[2], in_indices[2], self.S)
                else:
                    _filter[
                        out_indices,
                        in_indices,
                        :,
                    ] = expanded.reshape(-1, self.S)

        # return the new filter
        return _filter
