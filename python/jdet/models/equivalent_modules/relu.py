from jdet.utils.equivalent.geometric_tensor import GeometricTensor
from jittor import nn
from jdet.utils.equivalent import FieldType, GeneralOnR2

__all__ = ["eReLU"]

class eReLU(nn.Module):
    def __init__(self, in_type: FieldType, inplace: bool = False):
        r"""
        Module that implements a pointwise ReLU to every channel independently.
        The input representation is preserved by this operation and, therefore, it equals the output
        representation.
        Only representations supporting pointwise non-linearities are accepted as input field type.
        Args:
            in_type (FieldType):  the input field type
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``
            
        """
        assert isinstance(in_type.gspace, GeneralOnR2)
        if inplace:
            raise NotImplementedError
        
        super(eReLU, self).__init__()
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)
        self.space = in_type.gspace
        self.in_type = in_type
        # the representation in input is preserved
        self.out_type = in_type
        self._inplace = inplace
    
    def execute(self, *args, **kw) -> None:
        return super().execute(*args, **kw)

    def execute(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Applies ReLU function on the input fields
        Args:
            input (GeometricTensor): the input feature map
        Returns:
            the resulting feature map after relu has been applied
        """
        
        assert input.type == self.in_type, "Error! the type of the input does not match the input type of this module"
        return GeometricTensor(nn.relu(input.tensor), self.out_type)
