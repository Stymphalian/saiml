import numpy as np
import autograd2 as ag
from typing import *

def _unpack_params(value: object) -> List[ag.Tensor]:
    if isinstance(value, ag.Tensor):
        return [value]
    elif isinstance(value, Module):
        return value.unpack_params()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []
    
def _pack_params(value: object, flat_params: np.ndarray):
    if isinstance(value, ag.Tensor):
        value._data = flat_params.reshape(value.shape)
    elif isinstance(value, Module):
        value.set_params(flat_params)
    elif isinstance(value, dict):
        count = 0
        for k, v in value.items():
            v_size = sum([x.size for x in _unpack_params(v)])
            _pack_params(v, flat_params[count:count+v_size])
            count += v_size
    elif isinstance(value, (list, tuple)):
        count = 0
        for v in value:
            v_size = sum([x.size for x in _unpack_params(v)])
            _pack_params(v, flat_params[count:count+v_size])
            count += v_size

class Module:
    def __init__(self):
        # Sub-classes can set self.params and then we can automatically 
        # return the parameters and gradients for the module
        self.params = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        """Inference"""
        raise NotImplementedError()

    def backward(self, context):
        """Run backwards to update the gradients of all the parameters"""
        learning_rate = context["learning_rate"]
        for param in self.params:
            if isinstance(param, ag.Tensor):
                # TODO: How to integrate optimizer updates into this?
                param._data -= learning_rate * param.grad.value()
            elif isinstance(param, Module):
                param.backward(context)

    def unpack_params(self):
        return _unpack_params(self.params)

    def get_params(self):
        params = self.unpack_params()
        flat_params = [p.value().reshape(-1) for p in params]
        return np.concatenate(flat_params)
    
    def get_grads(self):
        params = self.unpack_params()
        grads = [x.grad.value().reshape(-1) for x in params]
        return np.concatenate(grads)
    
    def get_params_and_grads(self):
        params = self.unpack_params()
        flat_params = [p.value().reshape(-1) for p in params]
        grads = [x.grad.value().reshape(-1) for x in params]
        return (np.concatenate(flat_params), np.concatenate(grads))
    
    def set_params(self, flat_params):
        _pack_params(self.params, flat_params)