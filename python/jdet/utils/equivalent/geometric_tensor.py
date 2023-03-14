from __future__ import annotations
import jittor as jt
from .field_type import FieldType
from typing import Union

class GeometricTensor:
    
    def __init__(self, tensor: jt.Var, type: FieldType):
        r"""
        A GeometricTensor can be interpreted as a *typed* tensor.
        It is wrapping a common :class:`torch.Tensor` and endows it with a (compatible) :class:`~e2cnn.nn.FieldType` as
        *transformation law*.

        The :class:`~e2cnn.nn.FieldType` describes the action of a group :math:`G` on the tensor.
        This action includes both a transformation of the base space and a transformation of the channels according to
        a :math:`G`-representation :math:`\rho`.
        
        All *e2cnn* neural network operations have :class:`~e2cnn.nn.GeometricTensor` s as inputs and outputs.
        They perform a dynamic typechecking, ensuring that the transformation laws of the data and the operation match.
        See also :class:`~e2cnn.nn.EquivariantModule`.
 
        As usual, the first dimension of the tensor is interpreted as the batch dimension. The second is the fiber
        (or channel) dimension, which is associated with a group representation by the field type. The following
        dimensions are the spatial dimensions (like in a conventional CNN).
        
        The operations of **addition** and **scalar multiplication** are supported.
        For example::
            gs = e2cnn.gspaces.Rot2dOnR2(8)
            type = e2cnn.nn.FieldType(gs, [gs.regular_repr]*3)
            t1 = e2cnn.nn.GeometricTensor(torch.randn(1, 24, 3, 3), type)
            t2 = e2cnn.nn.GeometricTensor(torch.randn(1, 24, 3, 3), type)
            
            # addition
            t3 = t1 + t2
            
            # scalar product
            t3 = t1 * 3.
            
            # scalar product also supports tensors containing only one scalar
            t3 = t1 * torch.tensor(3.)
            
            # inplace operations are also supported
            t1 += t2
            t2 *= 3.
        
        .. warning ::
            The multiplication of a PyTorch tensor containing only a scalar with a GeometricTensor is only supported
            when using PyTorch 1.4 or higher (see this `issue <https://github.com/pytorch/pytorch/issues/26333>`_ )
            
        A GeometricTensor supports **slicing** in a similar way to PyTorch's :class:`torch.Tensor`.
        More precisely, slicing along the batch (1st) and the spatial (3rd, 4th, ...) dimensions works as usual.
        However, slicing the fiber (2nd) dimension would break equivariance when splitting channels belonging to
        the same field.
        To prevent this, slicing on the second dimension is defined over *fields* instead of channels.
        
        .. warning ::
            
            GeometricTensor only supports basic *slicing* but it does **not** support *advanced indexing* (see NumPy's
            documentation about
            `indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing>`_
            for more details).
            Moreover, in contrast to NumPy and PyTorch, an index containing a single integer value **does not** reduce
            the dimensionality of the tensor.
            In this way, the resulting tensor can always be interpreted as a GeometricTensor.
            
        
        We give few examples to illustrate this behavior::
        
            # Example of GeometricTensor slicing
            space = e2cnn.gspaces.Rot2dOnR2(4)
            type = e2cnn.nn.FieldType(space, [
                # field type            # index # size
                space.regular_repr,     #   0   #  4
                space.regular_repr,     #   1   #  4
                space.irrep(1),         #   2   #  2
                space.irrep(1),         #   3   #  2
                space.trivial_repr,     #   4   #  1
                space.trivial_repr,     #   5   #  1
                space.trivial_repr,     #   6   #  1
            ])                          #   sum = 15
            
            # this FieldType contains 8 fields
            len(type)
            >> 7
            
            # the size of this FieldType is equal to the sum of the sizes of each of its fields
            type.size
            >> 15
            
            geom_tensor = e2cnn.nn.GeometricTensor(torch.randn(10, type.size, 9, 9), type)
            
            geom_tensor.shape
            >> torch.Size([10, 15, 9, 9])
            
            geom_tensor[1:3, :, 2:5, 2:5].shape
            >> torch.Size([2, 15, 3, 3])
            
            geom_tensor[..., 2:5].shape
            >> torch.Size([10, 15, 9, 3])
            
            # the tensor contains the fields 1:4, i.e 1, 2 and 3
            # these fields have size, respectively, 4, 2 and 2
            # so the resulting tensor has 8 channels
            geom_tensor[:, 1:4, ...].shape
            >> torch.Size([10, 8, 9, 9])
            
            # the tensor contains the fields 0:6:2, i.e 0, 2 and 4
            # these fields have size, respectively, 4, 2 and 1
            # so the resulting tensor has 7 channels
            geom_tensor[:, 0:6:2].shape
            >> torch.Size([10, 7, 9, 9])
            
            # the tensor contains only the field 2, which has size 2
            # note, also, that even though a single index is used for the batch dimension, the resulting tensor
            # still has 4 dimensions
            geom_tensor[3, 2].shape
            >> torch.Size(1, 2, 9, 9)
        
        .. warning ::
        
            *Slicing* over the fiber (2nd) dimension with ``step > 1`` or with a negative step is converted
            into *indexing* over the channels.
            This means that, in these cases, slicing behaves like *advanced indexing* in PyTorch and NumPy
            **returning a copy instead of a view**.
            For more details, see the *note* `here <https://pytorch.org/docs/stable/tensor_view.html>`_ and
            *NumPy*'s `docs <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_ .
        
        .. note ::
    
            Slicing is not supported for setting values inside the tensor
            (i.e. :meth:`~object.__setitem__` is not implemented).
            Indeed, depending on the values which are assigned, this operation can break the symmetry of the tensor
            which may not transform anymore according to its transformation law (specified by ``type``).
            In case this feature is necessary, one can directly access the underlying :class:`torch.Tensor`, e.g.
            ``geom_tensor.tensor[:3, :, 2:5, 2:5] = torch.randn(3, 4, 3, 3)``, although this is not recommended.
            
            
        Args:
            tensor (torch.Tensor): the tensor data
            type (FieldType): the type of the tensor, modeling its transformation law
        
        Attributes:
            ~.tensor (torch.Tensor)
            ~.type (FieldType)
            
        """
        assert isinstance(tensor, jt.Var)
        assert isinstance(type, FieldType)
        
        assert len(tensor.shape) >= 2
        assert tensor.shape[1] == type.size, \
            f"Error! The size of the tensor {tensor.shape} does not match the size of the field type {type.size}."
        
        # torch.Tensor: PyTorch tensor containing the data
        self.tensor = tensor
        
        # FieldType: field type of the signal
        self.type = type

    def __add__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Add two compatible :class:`~e2cnn.nn.GeometricTensor` using pointwise addition.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            the sum

        """
        assert self.type == other.type, 'The two geometric tensor must have the same FieldType'
        return GeometricTensor(self.tensor + other.tensor, self.type)

    def __sub__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Subtract two compatible :class:`~e2cnn.nn.GeometricTensor` using pointwise subtraction.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            their difference

        """
        assert self.type == other.type, 'The two geometric tensor must have the same FieldType'
        return GeometricTensor(self.tensor - other.tensor, self.type)

    def __iadd__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Add a compatible :class:`~e2cnn.nn.GeometricTensor` to this tensor inplace.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            this tensor

        """
        assert self.type == other.type, 'The two geometric tensor must have the same FieldType'
        self.tensor += other.tensor
        return self

    def __isub__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Subtract a compatible :class:`~e2cnn.nn.GeometricTensor` to this tensor inplace.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            this tensor

        """
        assert self.type == other.type, 'The two geometric tensor must have the same FieldType'

        self.tensor -= other.tensor
        return self
    
    def __mul__(self, other: Union[float, jt.Var]) -> 'GeometricTensor':
        r"""
        Scalar product of this :class:`~e2cnn.nn.GeometricTensor` with a scalar.
        The operation is done inplace.
        
        The scalar can be a float number of a :class:`torch.Tensor` containing only
        one scalar (i.e. :func:`torch.numel` should return `1`).

        Args:
            other : a scalar

        Returns:
            the scalar product

        """
        assert isinstance(other, float) or other.numel() == 1, 'Only multiplication with a scalar is allowed'

        return GeometricTensor(self.tensor * other, self.type)

    __rmul__ = __mul__

    def __imul__(self, other: Union[float, jt.Var]) -> 'GeometricTensor':
        r"""
        Scalar product of this :class:`~e2cnn.nn.GeometricTensor` with a scalar.

        The scalar can be a float number of a :class:`torch.Tensor` containing only
        one scalar (i.e. :func:`torch.numel` should return `1`).

        Args:
            other : a scalar

        Returns:
            the scalar product

        """
        assert isinstance(other, float) or other.numel() == 1, 'Only multiplication with a scalar is allowed'
    
        self.tensor *= other
        return self
    
    def __repr__(self):
        t = repr(self.tensor)[:-1]
        t = t.replace('\n', '\n  ')
        r = 'g_' + t + ', ' + repr(self.type) + ')'

        return r
