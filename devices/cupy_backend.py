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

def memory_used():
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()

def print_memory(title):
    mempool = cp.get_default_memory_pool()
    KB = 1024
    MB = 1024 * KB
    print(title)
    print("  Memory Used: {:.2f} MB".format(mempool.used_bytes() / MB))
    print("  Memory Total: {:.2f} MB".format(mempool.total_bytes() / MB))
    print("  Memory Free: {:.2f} MB".format(mempool.free_bytes() / MB))
