from jittor import nn
from typing import Tuple, Union
from jdet.utils.equivalent.geometric_tensor import GeometricTensor
from jdet.utils.equivalent import FieldType, GeneralOnR2

__all__ = ["PointwiseAvgPool", "PointwiseMaxPool"]

class PointwiseAvgPool(nn.Module):
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""
        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool2D`, wrapping it in the
        :class:`~e2cnn.nn.EquivariantModule` interface.
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a average over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape

        """
        assert isinstance(in_type.gspace, GeneralOnR2)
        super(PointwiseAvgPool, self).__init__()
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.ceil_mode = ceil_mode

    def execute(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Args:
            input (GeometricTensor): the input feature map
        Returns:
            the resulting feature map
        """
        
        assert input.type == self.in_type
        
        # run the common max-pooling
        output = nn.avg_pool2d(input.tensor,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              ceil_mode=self.ceil_mode)
                
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)


class PointwiseMaxPool(nn.Module):
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 ceil_mode: bool = False
                 ):
        r"""
        Channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.MaxPool2D`, wrapping it in the
        :class:`~e2cnn.nn.EquivariantModule` interface.
        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when True, will use ceil instead of floor to compute the output shape

        """
        if dilation != 1 and dilation != (1, 1):
            raise NotImplementedError
        assert isinstance(in_type.gspace, GeneralOnR2)
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                f"""Error! Representation "{r.name}" does not support pointwise non-linearities
                so it is not possible to pool each channel independently"""
        super(PointwiseMaxPool, self).__init__()
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
        self.dilation = None
        self.ceil_mode = ceil_mode

    def execute(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        # run the common max-pooling
        output = nn.max_pool2d(input.tensor,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              ceil_mode=self.ceil_mode)
                
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

