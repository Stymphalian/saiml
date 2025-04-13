import autograd2 as ag

class Optimizer:
    def step(self, param: ag.Tensor,  grad: ag.Tensor):
        raise NotImplementedError()
    def batch_start(self, **kwargs):
        pass
    def batch_step(self, **kwargs):
        pass
    def batch_end(self, **kwargs):
        pass
    
class SGD(Optimizer):
    def __init__(self, lr=0.5):
        self.learning_rate = lr

    def step(self, param: ag.Tensor, grad: ag.Tensor):
        assert isinstance(param, ag.Tensor)
        assert isinstance(grad, ag.Tensor)
        return (param - self.learning_rate * grad).detach()
    

class SGDMomentum(Optimizer):
    def __init__(self, lr=5e-4, momentum=0.9):
        self.learning_rate = lr
        self.momentum = momentum  # beta
        self.prev_grads = {}

    def step(self, param: ag.Tensor, grad: ag.Tensor):
        assert isinstance(param, ag.Tensor)
        assert isinstance(grad, ag.Tensor)
        if param.id not in self.prev_grads:
            self.prev_grads[param.id] = ag.zeros(param.shape)

        prev = self.prev_grads[param.id]

        B = self.momentum
        Vt = B * prev + (1-B) * grad
        new_param = param - self.learning_rate * Vt

        self.prev_grads[param.id] = Vt.detach()
        return new_param
    
class RMSProp(Optimizer):
    def __init__(self, lr=5e-4, decay=0.9):
        self.learning_rate = lr
        self.decay = decay
        self.prev_grads = {}

    def step(self, param: ag.Tensor, grad: ag.Tensor):
        assert isinstance(param, ag.Tensor)
        assert isinstance(grad, ag.Tensor)
        if param.id not in self.prev_grads:
            self.prev_grads[param.id] = ag.zeros(param.shape)

        prev = self.prev_grads[param.id]

        D = self.decay
        Vt = D * prev + (1-D) * ag.power(grad,2)
        sqrt_vt = ag.sqrt(Vt) + 1e-8
        new_param = param - self.learning_rate * (grad / sqrt_vt)

        self.prev_grads[param.id] = Vt.detach()
        return new_param

class Adam(Optimizer):
    def __init__(self, lr=5e-4, momentum=0.9, decay=0.999):
        self.learning_rate = lr
        self.momentum = momentum
        self.decay = decay
        self.prev_velocity = {}
        self.prev_decay = {}
        self.iteration = 0
        self.lr_decay = 1

    def step(self, param: ag.Tensor, grad: ag.Tensor):
        assert isinstance(param, ag.Tensor)
        assert isinstance(grad, ag.Tensor)
        if param.id not in self.prev_velocity:
            self.prev_velocity[param.id] = ag.zeros(param.shape)
            self.prev_decay[param.id] = ag.zeros(param.shape)

        prev_v = self.prev_velocity[param.id]
        prev_d = self.prev_decay[param.id]

        M = self.momentum
        D = self.decay
        velocity = M * prev_v + (1-M) * grad
        decay = D * prev_d + (1-D) * ag.power(grad,2)

        velocity = velocity / (1 - (M**self.iteration))
        decay = decay / (1 - (D**self.iteration))

        sqrt_decay = ag.sqrt(decay) + 1e-8
        new_param = param - self.learning_rate * (velocity / sqrt_decay)

        self.prev_velocity[param.id] = velocity.detach()
        self.prev_decay[param.id] = decay.detach()
        return new_param
    
    def batch_start(self, **kwargs):
        epoch = kwargs["epoch"]
        self.learning_rate = (1 / (1 + self.lr_decay * epoch)) * self.learning_rate
        self.iteration = 1

    def batch_step(self, **kwargs):
        self.iteration += 1
    