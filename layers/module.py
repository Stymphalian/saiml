import numpy as np

class Module:
    def __init__(self):
        # Sub-classes can set self.params and then we can automatically 
        # return the parameters and gradients for the module
        self.params = []

    def forward(self, x):
        """Inference"""
        raise NotImplementedError()

    def backward(self, context):
        """Run backwards to update the gradients of all the parameters"""
        learning_rate = context["learning_rate"]
        for param in self.params:
            param._data -= learning_rate * param.grad

    def get_params_grads_size(self):
        params_size = sum(x.size for x in self.params)
        grads_size = sum(x.size for x in self.params)
        return (params_size, grads_size)
    
    def get_params_grads(self):
        params = [x._data.reshape(-1) for x in self.params]
        grads = [x.grad.reshape(-1) for x in self.params]
        if len(params) > 0:
            params = np.concatenate(params)
        else:
            params = np.array([])
        if len(grads) > 0:
            grads = np.concatenate(grads)
        else:
            grads = np.array([])
        return (params, grads)
    
    def set_params(self, params):
        count = 0
        for pi in range(len(self.params)):
            data_size = self.params[pi].size
            data_shape = self.params[pi].shape
            data = params[count:count+data_size].reshape(data_shape)
            self.params[pi]._data = data
            count += data_size