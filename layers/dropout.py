import autograd2 as ag
from devices import xp
from . import Module

# TODO: implement inverse dropout
class Dropout(Module):
    def __init__(self, is_training, p=0.5, rng_seed=None):
        super().__init__()
        self.p = p
        self.is_training = is_training
        self.rng_seed = rng_seed

    def forward(self, x):
        return ag.inverse_dropout(
            x,
            self.is_training,
            p=self.p,
            rng_seed=self.rng_seed
        )
        # if self.is_training:
        #     rand = xp.random.binomial(1, self.p, size=x.shape) 
        #     y = x * ag.Tensor(rand)
        #     return y
        # else:
        #     return x * self.p