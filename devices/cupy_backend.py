import cupy as cp
from .device import Device

class GPUDevice(Device):
    @property
    def array_datatype(self): 
        return cp.ndarray
    
    def array(self, *args, **kwargs):
        return cp.array(*args, **kwargs)
        
    def to_numpy(self, arr, *args, **kwargs):
        return cp.asnumpy(arr, *args, **kwargs)  

xp = cp
xp_ndarray = xp.ndarray

def default_device():
    return GPUDevice()
