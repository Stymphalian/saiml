class Device:
    def array(*args, **kwargs):
        raise NotImplementedError("Device not implemented")
    
    def from_numpy(self, *args, **kwargs):
        raise NotImplementedError("Device not implemented")
    
    @property
    def array_datatype(self): 
        return None