from .module import Module
import autograd2 as ag
from devices import xp

class Sampling(Module):
    def __init__(self, seed=1):
        super().__init__()
        self.seed = seed
        self.rng = xp.random.default_rng(seed)

    def forward(self, z_mean, z_var):
        assert z_mean.shape == z_var.shape
        dimensions = z_mean.size
        epsilon = self.rng.standard_normal(size=(dimensions, 1))
        return z_mean + ag.exp(0.5 * z_var) * epsilon