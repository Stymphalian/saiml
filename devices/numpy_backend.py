import numpy as np
from .device import Device

class CPUDevice(Device):
    @property
    def array_datatype(self): 
        return np.ndarray
    
    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)
    
    def to_numpy(self, arr, *args, **kwargs):
        return arr

xp = np
xp_ndarray = xp.ndarray

def default_device():
    return CPUDevice()
