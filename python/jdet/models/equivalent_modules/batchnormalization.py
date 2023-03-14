from __future__ import annotations
import jittor as jt
from jittor import nn
from jdet.utils.equivalent.equivalent_utils import indexes_from_labels, regular_feature_type
from jdet.utils.equivalent.geometric_tensor import GeometricTensor
from jdet.utils.equivalent import FieldType, GeneralOnR2

__all__ = ["InnerBatchNorm"]

class InnerBatchNorm(nn.Module):
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 ):
        r"""
        
        Batch normalization for representations with permutation matrices.
        Statistics are computed both over the batch and the spatial dimensions and over the channels within
        the same field (which are permuted by the representation).
        Only representations supporting pointwise non-linearities are accepted as input field type.
        
        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional):  if ``True``, this module has learnable affine parameters. Default: ``True``
            track_running_stats (bool, optional): when set to ``True``, the module tracks the running mean and variance;
                                                  when set to ``False``, it does not track such statistics but uses
                                                  batch statistics in both training and eval modes.
                                                  Default: ``True``
        """
        assert isinstance(in_type.gspace, GeneralOnR2)
        super(InnerBatchNorm, self).__init__()
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        # group fields by their size and
        #   - check if fields with the same size are contiguous
        #   - retrieve the indices of the fields
        grouped_fields = indexes_from_labels(self.in_type, [r.size for r in self.in_type.representations])

        # number of fields of each size
        self._nfields = {}
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = {}
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        for s, (contiguous, fields, indices) in grouped_fields.items():
            self._nfields[s] = len(fields)
            self._contiguous[s] = contiguous
            
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[s] = jt.array([min(indices), max(indices)+1], dtype=jt.int64)
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[s] = jt.array(indices, dtype=jt.int64)
                
            # register the indices tensors as parameters of this module
            # Different: original is self.register_buffer('indices_{}'.format(s), _indices[s])
            _indices[s].stop_grad()
            self.__setattr__('indices_{}'.format(s), _indices[s])
        
        for s in _indices.keys():
            _batchnorm = nn.BatchNorm3d(
                self._nfields[s],
                self.eps,
                self.momentum,
                affine=self.affine,
                # track_running_stats=self.track_running_stats
            )
            self.__setattr__('batch_norm_[{}]'.format(s), _batchnorm)
    
    def execute(self, input:GeometricTensor) -> GeometricTensor:
        r"""
        Args:
            input (GeometricTensor): the input feature map
        Returns:
            the resulting feature map
        """
        
        assert input.type == self.in_type
        b, c, h, w = input.tensor.shape
        # output = torch.empty_like(input.tensor)
        output = jt.zeros_like(input.tensor)
        
        # iterate through all field sizes
        for s, contiguous in self._contiguous.items():
            
            indices = getattr(self, f"indices_{s}")
            batchnorm = getattr(self, f'batch_norm_[{s}]')
            
            if contiguous:
                # if the fields were contiguous, we can use slicing
                output[:, indices[0].item():indices[1].item(), :, :] = batchnorm(
                    input.tensor[:, indices[0].item():indices[1].item(), :, :].view(b, -1, s, h, w)
                ).view(b, -1, h, w)
            else:
                # otherwise we have to use indexing
                output[:, indices, :, :] = batchnorm(
                    input.tensor[:, indices, :, :].view(b, -1, s, h, w)
                ).view(b, -1, h, w)
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

