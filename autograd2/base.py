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
        Compute the 
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

        self.data = np.array(data, dtype=dtype)
        self.dtype = dtype
        self.operator = operator
        self.grad = None
        self.inputs = inputs
        self.requires_grad = requires_grad

    def value(self):
        if self.data is None:
            if self.operator is not None:
                self.data = self.operator.compute(*self.inputs)
        return self.data
    
    def backward(self):
        ag.grad(self)

    def gradients(self, outGrad):
        if self.operator is None:
            return np.multiply(outGrad,np.ones(self.data.shape))
        return self.operator.gradients(self, outGrad)

    @property
    def shape(self):
        return self.value().shape
    @property
    def ndim(self):
        return self.value().ndim
    @property
    def size(self):
        return self.value().size
    
    def __repr__(self):
        return "Tensor(" + str(self.value()) + ")"

    def __str__(self):
        return self.value().__str__()
    
    def __add__(self, other):
        return ag.operators.add(self, other)
    
    def __sub__(self, other):
        return ag.operators.sub(self, other)

    def __mul__(self, other):
        return ag.operators.mult(self, other)
    
    def __truediv__(self, other):
        return ag.operators.div(self, other)
    
    