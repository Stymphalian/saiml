import numpy as np
from typing import *
from collections import defaultdict
import autograd2 as ag

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

    def gradients(self, node: "Tensor", outGrad: any) -> Tuple[Union["Tensor","TensorTuple"]]:
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
    
    def tensor_tuple(self, *inputs):
        data = self.compute(*inputs)
        return TensorTuple(data, operator=self, inputs=inputs)

class Tensor(Node):
    def __init__(
            self,
            data, 
            operator: Optional[Operator]=None,
            requires_grad=False,
            dtype=np.float64, 
            inputs: Tuple["Tensor"]=()):
        super().__init__()

        if isinstance(data, (Tensor, TensorTuple, np.ndarray)):
            self._data = data
        else:
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

    @property
    def grad(self):
        return self._grad
    
    @grad.setter
    def grad(self, value):
        self._grad = value
        # if self.grad is None:
        #     self._grad = value
        # else:
        #     self._grad[:] = value
    
    def backward(self, outGrad=None):
        ag.grad(self, outGrad=outGrad)

    def gradients(self, outGrad):
        if self.operator is None:
            return ag.mult(outGrad, Tensor(np.ones(self._data.shape)))
            # return np.multiply(outGrad,np.ones(self._data.shape))
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
        return "Tensor(" + self.value().__str__() + ")"
    
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
    # __rtruediv__ = __truediv__
        
    def reshape(self, shape):
        return ag.math_ops.reshape(self, shape)
    
    def ones(self):
        return Tensor(np.ones(self.shape), dtype=self.dtype)
    def zeros(self):
        return Tensor(np.zeros(self.shape), dtype=self.dtype)
    def random(self):
        return Tensor(np.random.rand(*self.shape), dtype=self.dtype)
    def arange(self):
        return Tensor(np.arange(self.shape[0]), dtype=self.dtype)
    
    @property
    def T(self):
        return ag.math_ops.transpose(self)

class TensorTuple(Node):
    def __init__(
            self,
            data,
            operator: Optional[Operator]=None,
            requires_grad=False,
            dtype=np.float64, 
            inputs: Tuple["Tensor"]=()):
        super().__init__()

        assert isinstance(data, (tuple, list))
        assert len(data) > 0
        if isinstance(data[0], Tensor):
            self._data = data
        else:
            self._data = [Tensor(x, dtype=x.dtype) for x in data]

        self._grad = None
        self.dtype = dtype
        self.operator = operator
        self.inputs = inputs
        self.requires_grad = requires_grad

    def value(self):
        if self._data is None and self.operator is not None:
            self._data = self.operator.compute(*self.inputs)
        return self._data
    
    def backward(self, outGrad=None):
        ag.grad(self, outGrad=outGrad)

    def gradients(self, outGrad):
        if self.operator is None:
            # TODO: This should be a tensor
            return ag.mult(outGrad, Tensor(np.ones(self._data.shape)))
            # return np.multiply(outGrad, np.ones(self._data.shape))
        grads = self.operator.gradients(self, outGrad)
        if type(grads) is not tuple:
            grads = (grads,)
        assert len(grads) == len(self.inputs)
        return grads
    
    def __len__(self):
        return len(self.value())
    
    def __getitem__(self, index):
        # TODO: handle tuple indices a[1:4]
        return ag.tuple_get_item(self, index)
        # return ag.Tensor(self._data[index], requires_grad=self.requires_grad)

    def __add__(self, other):
        if isinstance(other, TensorTuple):
            assert len(self) == len(other)
            return ag.make_tuple(*[self[i] + other[i] for i in range(len(self))])
        else:
            # TODO: Handle scalar
            return self
            # other = ag.make_tuple(*[Tensor(np.zeros(self[i].shape)) for i in range(len(self))])
            # return ag.make_tuple(*[self[i] + other[i] for i in range(len(self))])
    __radd__ = __add__
    
    def __repr__(self):
        return "TensorTuple(" + str(self.value()) + ")"
    
    def __str__(self):
        return "TensorTuple(" + self.value().__str__() + ")"
    
    

def grad(node: Union[Tensor, TensorTuple], outGrad=None):
    node_to_grads = defaultdict(list)

    if outGrad is None:
        if isinstance(node, TensorTuple):
            outGrad = TensorTuple([a.ones() for a in node])
        else:
            outGrad = node.ones()
        # outGrad = np.ones(node.value().shape, dtype=np.float64)
    # assert isinstance(outGrad, "Tensor")
    node_to_grads[node.id] = [outGrad]
    
    for v in ag.toposort(node):
        dy_dv = sum(node_to_grads[v.id])
        # assert dy_dv.shape == node_to_grads[v.id][0].shape
        
        if v.requires_grad:
            v.grad = dy_dv
        dy_dp = v.gradients(dy_dv)
        for pi, p in enumerate(v.inputs):
            node_to_grads[p.id].append(dy_dp[pi])        
    return node_to_grads

# TODO: Need to replace this sum with ag.summation?