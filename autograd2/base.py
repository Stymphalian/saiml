import autograd2 as ag
import numpy as np
from typing import *

class Node:
    _NODE_AUTO_ID = 1
    def __init__(self):
        self.id = Node._NODE_AUTO_ID
        Node._NODE_AUTO_ID += 1

    def value(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
    
class Operator:
    def compute(self, *inputs: Tuple["Tensor"]):
        """
        Compute the value in the forward pass
        """
        raise NotImplementedError

    def gradients(self, node: "Tensor", outGrad: any) -> Tuple["Tensor"]:
        """
        Implement this function to compute the gradient of the operator.
        It should return a tuple with the same number of elements as the number
        of inputs used in the compute() method.
        """
        raise NotImplementedError
    
    def tensor(self, *inputs):
        """
        Return a new tensor which uses this operator to compute its value/gradient.
        """
        data = self.compute(*inputs)
        return Tensor(data, operator=self, inputs=inputs)

class Tensor(Node):
    def __init__(
            self,
            data, 
            operator: Optional[Operator]=None,
            requires_grad=False,
            dtype=np.float64, 
            inputs: Tuple["Tensor"]=()):
        super().__init__()

        self._data = np.array(data, dtype=dtype)
        self._grad = None
        self.dtype = dtype
        self.operator = operator
        self.inputs = inputs
        self.requires_grad = requires_grad

    def value(self):
        if self._data is None:
            if self.operator is not None:
                self._data = self.operator.compute(*self.inputs)
        return self._data
    
    # @property
    # def data(self):
    #     return self.value()
    
    # @data.setter
    # def data(self, value):
    #     self.value()
    #     self._data[:] = value

    @property
    def grad(self):
        return self._grad
    
    @grad.setter
    def grad(self, value):
        if self.grad is None:
            self._grad = value
        else:
            self._grad[:] = value
    
    def backward(self):
        ag.grad(self)

    def gradients(self, outGrad):
        if self.operator is None:
            return np.multiply(outGrad,np.ones(self._data.shape))
        grads = self.operator.gradients(self, outGrad)
        if type(grads) is not tuple:
            grads = (grads,)
        assert len(grads) == len(self.inputs)
        return grads

    @property
    def shape(self):
        return self.value().shape
    @property
    def ndim(self):
        return self.value().ndim
    @property
    def size(self):
        return self.value().size
    
    def __getitem__(self, index):
        return ag.Tensor(self._data[index], requires_grad=self.requires_grad)

    def __setitem__(self, index, value):
        if not isinstance(value, Tensor):
            self._data[index] = value
        else:
            self._data[index] = value.value()
    
    def __repr__(self):
        return "Tensor(" + str(self.value()) + ")"
    
    def __str__(self):
        return self.value().__str__()
    
    def __neg__(self):
        return ag.math_ops.neg(self)
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return ag.math_ops.add(self, other)
        else:
            return ag.math_ops.add_scalar(self, other)
    __radd__ = __add__
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return ag.math_ops.sub(self, other)
        else:
            return ag.math_ops.sub_scalar(self, other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return ag.math_ops.mult(self, other)
        else:
            return ag.math_ops.mult_scalar(self, other)
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return ag.math_ops.div(self, other)
        else:
            return ag.math_ops.div_scalar(self, other)
    
    