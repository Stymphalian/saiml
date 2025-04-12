use_cpu = True
from .device import *

if use_cpu:
    from .numpy_backend import default_device
    from .numpy_backend import xp
    from .numpy_backend import xp_ndarray
else:
    from .cupy_backend import default_device
    from .cupy_backend import xp
    from .cupy_backend import xp_ndarray