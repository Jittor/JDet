from jdet.utils.equivalent import FieldType, GeneralOnR2, regular_feature_type, GeometricTensor
from typing import Union, List, Callable, Tuple
from jittor import nn
from collections import defaultdict
from .block_basis import BlocksBasisExpansion
import jittor as jt
import numpy as np
import math

__all__ = ["R2Conv"]

class R2Conv(nn.Module):
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 groups: int = 1,
                 bias: bool = True,
                 basisexpansion: str = 'blocks',
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 ):
        r"""
        G-steerable planar convolution mapping between the input and output :class:`~e2cnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^2\rtimes G` where :math:`G` is the
        :attr:`e2cnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.
        
        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~e2cnn.nn.R2Conv` guarantees an equivariant mapping
        
        .. math::
            \kappa \star [\mathcal{T}^\text{in}_{g,u} . f] = \mathcal{T}^\text{out}_{g,u} . [\kappa \star f] \qquad\qquad \forall g \in G, u \in \R^2
            
        where the transformation of the input and output fields are given by
 
        .. math::
            [\mathcal{T}^\text{in}_{g,u} . f](x) &= \rho_\text{in}(g)f(g^{-1} (x - u)) \\
            [\mathcal{T}^\text{out}_{g,u} . f](x) &= \rho_\text{out}(g)f(g^{-1} (x - u)) \\

        The equivariance of G-steerable convolutions is guaranteed by restricting the space of convolution kernels to an
        equivariant subspace.
        As proven in `3D Steerable CNNs <https://arxiv.org/abs/1807.02547>`_, this parametrizes the *most general
        equivariant convolutional map* between the input and output fields.
        For feature fields on :math:`\R^2` (e.g. images), the complete G-steerable kernel spaces for :math:`G \leq \O2`
        is derived in `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_.

        During training, in each forward pass the module expands the basis of G-steerable kernels with learned weights
        before calling :func:`torch.nn.functional.conv2d`.
        When :meth:`~torch.nn.Module.eval()` is called, the filter is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the kernel remains.
        
        .. warning ::
            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~e2cnn.nn.R2Conv.filter` and
            :attr:`~e2cnn.nn.R2Conv.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`e2cnn.nn.R2Conv.train`.
            
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
 
        The learnable expansion coefficients of the this module can be initialized with the methods in
        :mod:`e2cnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~e2cnn.nn.init.generalized_he_init`.
        
        .. warning ::
            
            This initialization procedure can be extremely slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~e2cnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.
        
        
        The parameters ``basisexpansion``, ``sigma``, ``frequencies_cutoff``, ``rings`` and ``maximum_offset`` are
        optional parameters used to control how the basis for the filters is built, how it is sampled on the filter
        grid and how it is expanded to build the filter. We suggest to keep these default values.

        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            kernel_size (int): the size of the (square) filter
            padding (int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            padding_mode(str, optional): ``zeros``, ``reflect``, ``replicate`` or ``circular``. Default: ``zeros``
            stride (int, optional): the stride of the kernel. Default: ``1``
            dilation (int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            basisexpansion (str, optional): the basis expansion algorithm to use
            sigma (list or float, optional): width of each ring where the bases are sampled. If only one scalar
                    is passed, it is used for all rings.
            frequencies_cutoff (callable or float, optional): function mapping the radii of the basis elements to the
                    maximum frequency accepted. If a float values is passed, the maximum frequency is equal to the
                    radius times this factor. By default (``None``), a more complex policy is used.
            rings (list, optional): radii of the rings where to sample the bases
            maximum_offset (int, optional): number of additional (aliased) frequencies in the intertwiners for finite
                    groups. By default (``None``), all additional frequencies allowed by the frequencies cut-off
                    are used.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``
        
        Attributes:
            ~.weights (jt.Var): the learnable parameters which are used to expand the kernel
            ~.filter (jt.Var): the convolutional kernel obtained by expanding the parameters
                                    in :attr:`~e2cnn.nn.R2Conv.weights`
            ~.bias (jt.Var): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (jt.Var): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~e2cnn.nn.R2Conv.bias`
        
        """

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)

        super(R2Conv, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.groups = groups

        if isinstance(padding, tuple) and len(padding) == 2:
            _padding = padding
        elif isinstance(padding, int):
            _padding = (padding, padding)
        else:
            raise ValueError('padding needs to be either an integer or a tuple containing two integers but {} found'.format(padding))
        
        padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in padding_modes:
            raise ValueError("padding_mode must be one of [{}], but got padding_mode='{}'".format(padding_modes, padding_mode))
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(_padding) for _ in range(2))
        
        if groups > 1:
            # Check the input and output classes can be split in `groups` groups, all equal to each other
            # first, check that the number of fields is divisible by `groups`
            assert len(in_type) % groups == 0
            assert len(out_type) % groups == 0
            in_size = len(in_type) // groups
            out_size = len(out_type) // groups
            
            # then, check that all groups are equal to each other, i.e. have the same types in the same order
            assert all(in_type.representations[i] == in_type.representations[i % in_size] for i in range(len(in_type)))
            assert all(out_type.representations[i] == out_type.representations[i % out_size] for i in range(len(out_type)))
            
            # finally, retrieve the type associated to a single group in input.
            # this type will be used to build a smaller kernel basis and a smaller filter
            # as in PyTorch, to build a filter for grouped convolution, we build a filter which maps from one input
            # group to all output groups. Then, PyTorch's standard convolution routine interpret this filter as `groups`
            # different filters, each mapping an input group to an output group.
            in_type = in_type.index_select(list(range(in_size)))
        
        if bias:
            # bias can be applied only to trivial irreps inside the representation
            # to apply bias to a field we learn a bias for each trivial irreps it contains
            # and, then, we transform it with the change of basis matrix to be able to apply it to the whole field
            # this is equivalent to transform the field to its irreps through the inverse change of basis,
            # sum the bias only to the trivial irrep and then map it back with the change of basis
            
            # count the number of trivial irreps
            trivials = 0
            for r in self.out_type:
                for irr in r.irreps:
                    if self.out_type.fibergroup.irreps[irr].is_trivial():
                        trivials += 1
            
            # if there is at least 1 trivial irrep
            if trivials > 0:
                
                # matrix containing the columns of the change of basis which map from the trivial irreps to the
                # field representations. This matrix allows us to map the bias defined only over the trivial irreps
                # to a bias for the whole field more efficiently
                bias_expansion = jt.zeros((self.out_type.size, trivials))
                
                p, c = 0, 0
                for r in self.out_type:
                    pi = 0
                    for irr in r.irreps:
                        irr = self.out_type.fibergroup.irreps[irr]
                        if irr.is_trivial():
                            bias_expansion[p:p+r.size, c] = jt.array(r.change_of_basis[:, pi])
                            c += 1
                        pi += irr.size
                    p += r.size
                
                # Different: self.register_buffer("bias_expansion", bias_expansion)
                self.bias_expansion = bias_expansion
                self.bias_expansion.stop_grad()
                # Different: self.bias = Parameter(torch.zeros(trivials), requires_grad=True), register_buffer()
                self.bias = jt.zeros(trivials)
                self.expanded_bias = jt.zeros(out_type.size)
                self.expanded_bias.stop_grad()          
            else:
                self.bias = None
                self.expanded_bias = None
        else:
            self.bias = None
            self.expanded_bias = None

        grid, basis_filter, rings, sigma, maximum_frequency = compute_basis_params(kernel_size,
                                                                                   frequencies_cutoff,
                                                                                   rings,
                                                                                   sigma,
                                                                                   dilation,
                                                                                   basis_filter)
        
        # BasisExpansion: submodule which takes care of building the filter
        self._basisexpansion = None
        
        # notice that `in_type` is used instead of `self.in_type` such that it works also when `groups > 1`
        if basisexpansion == 'blocks':
            self._basisexpansion = BlocksBasisExpansion(in_type, out_type,
                                                        basis_generator=self.space.build_kernel_basis,
                                                        points=grid,
                                                        sigma=sigma,
                                                        rings=rings,
                                                        maximum_offset=maximum_offset,
                                                        maximum_frequency=maximum_frequency,
                                                        basis_filter=basis_filter,
                                                        recompute=recompute)

        else:
            raise ValueError('Basis Expansion algorithm "%s" not recognized' % basisexpansion)
        
        if self.basisexpansion.dimension() == 0:
            raise ValueError('''
                The basis for the steerable filter is empty!
                Tune the `frequencies_cutoff`, `kernel_size`, `rings`, `sigma` or `basis_filter` parameters to allow
                for a larger basis.
            ''')

        self.weights = jt.zeros(self.basisexpansion.dimension())
        # self.register_buffer("filter", torch.zeros(out_type.size, in_type.size, kernel_size, kernel_size))
        self.filter = jt.zeros((out_type.size, in_type.size, kernel_size, kernel_size))
        self.filter.stop_grad()
        
        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            generalized_he_init(self.weights, self.basisexpansion)

    @property
    def basisexpansion(self) -> BlocksBasisExpansion:
        r"""
        Submodule which takes care of building the filter.
        
        It uses the learnt ``weights`` to expand a basis and returns a filter in the usual form used by conventional
        convolutional modules.
        It uses the learned ``weights`` to expand the kernel in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in}, s^2)`, where :math:`s` is the ``kernel_size``.
        
        """
        return self._basisexpansion

    def expand_parameters(self) -> Tuple[jt.Var, jt.Var]:
        r"""
        Expand the filter in terms of the :attr:`e2cnn.nn.R2Conv.weights` and the
        expanded bias in terms of :class:`e2cnn.nn.R2Conv.bias`.
        Returns:
            the expanded filter and bias
        """
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)
        
        if self.bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion @ self.bias
            
        return _filter, _bias

    def execute(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded filter and bias.
        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``
        Returns:
            output feature field transforming according to ``out_type``
        """
        assert input.type == self.in_type
        # TODO(514flowey): faster evaluation
        # if not self.is_training():
        #     _filter = self.filter
        #     _bias = self.expanded_bias
        # else:
            # retrieve the filter and the bias
        _filter, _bias = self.expand_parameters()
        
        # use it for convolution and return the result
        if self.padding_mode == 'zeros':
            output = nn.conv2d(input.tensor, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups,
                            bias=_bias)
        else:
            output = nn.conv2d(nn.pad(input.tensor, self._reversed_padding_repeated_twice, self.padding_mode),
                            _filter,
                            stride=self.stride,
                            dilation=self.dilation,
                            padding=(0,0),
                            groups=self.groups,
                            bias=_bias)

        return GeometricTensor(output, self.out_type)


def _generalized_he_init_variances(basisexpansion: BlocksBasisExpansion):
    r"""
    Compute the variances of the weights of a convolutional layer with a generalized He's weight initialization method.
    Args:
        basisexpansion (BasisExpansion): the basis expansion method
    """
    vars = jt.ones((basisexpansion.dimension(),))
    inputs_count = defaultdict(lambda: set())
    basis_count = defaultdict(int)
    basis_info = list(basisexpansion.get_basis_info())

    for attr in basis_info:
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        inputs_count[o].add(in_irrep)
        basis_count[(in_irrep, o)] += 1
    
    for o in inputs_count.keys():
        inputs_count[o] = len(inputs_count[o])
    
    for w, attr in enumerate(basis_info):
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        vars[w] = 1. / math.sqrt(inputs_count[o] * basis_count[(in_irrep, o)])

    return vars

cached_he_vars = {}

def generalized_he_init(tensor: jt.Var, basisexpansion: BlocksBasisExpansion, cache: bool = False):
    r"""
    Initialize the weights of a convolutional layer with a generalized He's weight initialization method.
    Because the computation of the variances can be expensive, to save time on consecutive runs of the same model,
    it is possible to cache the tensor containing the variance of each weight, for a specific ```basisexpansion```.
    This can be useful if a network contains multiple convolution layers of the same kind (same input and output types,
    same kernel size, etc.) or if one needs to train the same network from scratch multiple times (e.g. to perform
    hyper-parameter search over learning rate or to repeat an experiment with different random seeds).
    .. note ::
        The variance tensor is cached in memory and therefore is only available to the current process.
    Args:
        tensor (torch.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method
        cache (bool, optional): cache the variance tensor. By default, ```cache=False```
    """
    assert tensor.shape == (basisexpansion.dimension(),)
    
    if cache and basisexpansion not in cached_he_vars:
        cached_he_vars[basisexpansion] = _generalized_he_init_variances(basisexpansion)
    
    if cache:
        vars = cached_he_vars[basisexpansion]
    else:
        vars = _generalized_he_init_variances(basisexpansion)

    tensor[:] = vars * jt.randn_like(tensor)

def get_grid_coords(kernel_size: int, dilation: int = 1):
    actual_size = dilation * (kernel_size -1) + 1
    origin = actual_size / 2 - 0.5
    points = []
    for y in range(kernel_size):
        y *= dilation
        for x in range(kernel_size):
            x *= dilation
            p = (x - origin, -y + origin)
            points.append(p)
    points = np.array(points)
    assert points.shape == (kernel_size ** 2, 2), points.shape
    return points.T

def _manual_fco3(max_radius: float) -> Callable[[float], float]:
    r"""
    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.
    Args:
        max_radius (float): radius of the last ring touching the border of the grid
    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else 1 if r == max_radius else 2
        return max_freq
    return bl_filter

def _manual_fco2(max_radius: float) -> Callable[[float], float]:
    r"""
    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.
    Args:
        max_radius (float): radius of the last ring touching the border of the grid
    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 1 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    return bl_filter

def _manual_fco1(max_radius: float) -> Callable[[float], float]:
    r"""
    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.
    Args:
        max_radius (float): radius of the last ring touching the border of the grid
    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 2 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    return bl_filter

def bandlimiting_filter(frequency_cutoff: Union[float, Callable[[float], float]]) -> Callable[[dict], bool]:
    r"""
    Returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (True) or not (False)
    If the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. In thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.
    If the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.
    Args:
        frequency_cutoff (float): factor controlling the bandlimiting
    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not
    """
    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff
    
    def bl_filter(attributes: dict) -> bool:
        return math.fabs(attributes["frequency"]) <= frequency_cutoff(attributes["radius"])
    
    return bl_filter

def compute_basis_params(kernel_size: int,
                         frequencies_cutoff: Union[float, Callable[[float], float]] = None,
                         rings: List[float] = None,
                         sigma: List[float] = None,
                         dilation: int = 1,
                         custom_basis_filter: Callable[[dict], bool] = None,
                         ):
    
    # compute the coordinates of the centers of the cells in the grid where the filter is sampled
    grid = get_grid_coords(kernel_size, dilation)
    max_radius = np.sqrt((grid **2).sum(0)).max()

    # by default, the number of rings equals half of the filter size
    if rings is None:
        n_rings = math.ceil(kernel_size / 2)
        rings = jt.linspace(0, (kernel_size - 1) // 2, n_rings) * dilation
        rings = rings.tolist()
    
    assert all([max_radius >= r >= 0 for r in rings])
    
    if sigma is None:
        sigma = [0.6] * (len(rings) - 1) + [0.4]
        for i, r in enumerate(rings):
            if r == 0.:
                sigma[i] = 0.005
                
    elif isinstance(sigma, float):
        sigma = [sigma] * len(rings)
        
    # TODO - use a string name for this setting
    if frequencies_cutoff is None:
        frequencies_cutoff = -1.
    
    if isinstance(frequencies_cutoff, float):
        if frequencies_cutoff == -3:
            frequencies_cutoff = _manual_fco3(kernel_size // 2)
        elif frequencies_cutoff == -2:
            frequencies_cutoff = _manual_fco2(kernel_size // 2)
        elif frequencies_cutoff == -1:
            frequencies_cutoff = _manual_fco1(kernel_size // 2)
        else:
            frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r
    
    # check if the object is a callable function
    assert callable(frequencies_cutoff)
    
    maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

    fco_filter = bandlimiting_filter(frequencies_cutoff)

    if custom_basis_filter is not None:
        basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (custom_basis_filter(d) and fco_filter(d))
    else:
        basis_filter = fco_filter
    
    return grid, basis_filter, rings, sigma, maximum_frequency
