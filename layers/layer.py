import numpy as np

from loss import *
import utils

class Layer:
    def __init__(self, frozen=False):
        self.frozen = frozen
    def getParameters(self):
        return []
    def setParameters(self, parameters):
        return
    def getGradients(self):
        return []
    def forward(self, context, X):
        pass
    def backward(self, context, dE):
        pass
