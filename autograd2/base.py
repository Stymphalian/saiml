import numpy as np
import graphviz
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
    def backward(self, outGrad=None):
        raise NotImplementedError
    def gradients(self, outGrad: Union["Tensor", "TensorTuple"]):
        raise NotImplementedError
    
class Operator:
    def compute(self, *inputs: Tuple["Tensor"]):
        """
        Compute the value in the forward pass
        """
        raise NotImplementedError
    
    def __repr__(self):
        return self.__class__.__name__
    def __str__(self):
        return self.__repr__()

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
            name="",
            inputs: Tuple["Tensor"]=()):
        super().__init__()

        if not isinstance(data, np.ndarray):
            self._data = np.array(data, dtype=dtype)
        else:
            assert isinstance(data, np.ndarray)
            self._data = data
        # if isinstance(data, (Tensor, TensorTuple, np.ndarray)):
        #     self._data = data
        # else:
        #     self._data = np.array(data, dtype=dtype)

        self._grad = None
        self.dtype = dtype
        self.operator = operator
        self.inputs = inputs
        self.requires_grad = requires_grad
        self._name = name

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

    @property 
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def set_name(self, value):
        self._name = value
        return self
    
    def backward(self, outGrad=None):
        ag.grad(self, outGrad=outGrad)
        return self.grad

    def gradients(self, outGrad) -> Tuple[any]:
        if self.operator is None:
            return (ag.mult(outGrad, self.ones()),)

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
        if isinstance(value, Tensor):
            self._data[index] = value.value()[index]
        else:
            self._data[index] = value
    
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
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return ag.math_ops.sub(other, self)
        else:
            return ag.math_ops.sub(ag.constant(other), self)

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
    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return ag.math_ops.div(other, self)
        else:
            return ag.math_ops.div(ag.constant(other), self)
        
    def __or__(self, other):
        return ag.math_ops.bitwise_or(self, other)
        
    @property
    def T(self):
        return self.transpose()
    def transpose(self):
        return ag.math_ops.transpose(self)
        
    def reshape(self, shape):
        return ag.math_ops.reshape(self, shape)
    def flatten(self):
        return ag.math_ops.reshape(self, (-1,))

    def ones(self):
        return ones(self.shape, dtype=self.dtype)
    def zeros(self):
        return zeros(self.shape, dtype=self.dtype)
    def random(self):
        return random(self.shape, dtype=self.dtype)
    def arange(self):
        return arange(self.shape, dtype=self.dtype)

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
        self._data = data
        self._grad = None
        self.dtype = dtype
        self.operator = operator
        self.inputs = inputs
        self.requires_grad = requires_grad
        self._name = ""

    def value(self):
        if self._data is None and self.operator is not None:
            self._data = self.operator.compute(*self.inputs)
        return self._data
    
    def backward(self, outGrad=None):
        ag.grad(self, outGrad=outGrad)

    def gradients(self, outGrad):
        assert self.operator is not None
        grads = self.operator.gradients(self, outGrad)
        if type(grads) is not tuple:
            grads = (grads,)
        assert len(grads) == len(self.inputs)
        return grads
    
    @property 
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def set_name(self, value):
        self._name = value
        return self
    
    def __len__(self):
        return len(self.value())
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return ag.tuple_get_slice(self, index)
        else:
            return ag.tuple_get_item(self, index)

    def __add__(self, other):
        if isinstance(other, TensorTuple):
            assert len(self) == len(other)
            return ag.make_tuple(*[self[i] + other[i] for i in range(len(self))])
        else:
            scalar = ag.Tensor(other)
            return ag.make_tuple(*[self[i] + scalar for i in range(len(self))])
    __radd__ = __add__
    
    def __repr__(self):
        return "TensorTuple(" + str(self.value()) + ")"
    
    def __str__(self):
        return "TensorTuple(" + self.value().__str__() + ")"
    
def ones(shape, dtype=np.float64, requires_grad=False):
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad)
def zeros(shape, dtype=np.float64, requires_grad=False):
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad)
def random(shape, dtype=np.float64, requires_grad=False):
    return Tensor(np.random.rand(*shape), dtype=dtype, requires_grad=requires_grad)
def arange(shape, dtype=np.float64, requires_grad=False):
    size = np.prod(shape)
    return Tensor(np.arange(size).reshape(shape), dtype=dtype, requires_grad=requires_grad)

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
        if len(node_to_grads[v.id]) == 1:
            dy_dv = node_to_grads[v.id][0]
        else:
            if isinstance(node_to_grads[v.id][0], TensorTuple):
                dy_dv = ag.tuple_sum(*node_to_grads[v.id])
            else:
                dy_dv = sum(node_to_grads[v.id])
        # assert dy_dv.shape == node_to_grads[v.id][0].shape
        
        if v.requires_grad:
            v.grad = dy_dv
        dy_dp = v.gradients(dy_dv)
        # assert len(dy_dp) == len(v.inputs), f"{len(dy_dp)} != {len(v.inputs)}"

        for pi, p in enumerate(v.inputs):
            node_to_grads[p.id].append(dy_dp[pi])        
    return node_to_grads

class Parameter(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad = True

class Gradient(Tensor):
    pass 


def generate_graphviz(node: Tensor):
    dot = graphviz.Digraph()
    for v in ag.toposort(node):


        attrs = {
            "shape": "box"
        }
        labels = []
        if v.name:
            labels.append(v.name)
        if v.operator:
            labels.append(f"{v.operator}")
        if isinstance(v, TensorTuple):
            labels.append(f"{v.__class__.__name__}")
            attrs["color"] = "blue"
        if isinstance(v, Tensor):
            if (v.size <= 4):
                labels.append(f"{v.value()}")
            else:
                labels.append(f"{v.value().shape}")
        label = "\n".join(labels)
        if isinstance(v, Parameter):
            attrs["shape"] = "doublecircle"

        if isinstance(v, Tensor):
            tooltip_value = str(np.round(v.value(),2))
        else:
            tooltip_value = str(v.value())
        attrs["tooltip"] = tooltip_value
        
        dot.node(str(v.id), label, **attrs)
        for y in v.inputs:
            dot.edge(str(y.id), str(v.id))
    return dot

def render_graphviz(node: Tensor):
    dot = generate_graphviz(node)
    dot.render("graphviz", view=True, format="svg")