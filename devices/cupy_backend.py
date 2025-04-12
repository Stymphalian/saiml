import cupy as cp
from .device import Device

class GPUDevice(Device):
    @property
    def array_datatype(self): 
        return cp.ndarray
    
    def array(self, *args, **kwargs):
        return cp.array(*args, **kwargs)
    
    def from_numpy(self, *args, **kwargs):
        return cp.array(*args, **kwargs)

xp = cp
xp_ndarray = xp.ndarray

def default_device():
    return GPUDevice()
