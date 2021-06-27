"""Quanfima initialization."""

import logging

__version__ = '1.0'
__log_name__ = 'quanfima'
log = logging.getLogger(__log_name__)

cuda_available = True
try:
    import pycuda
except ImportError:
    cuda_available = False

visvis_available = True
try:
    import visvis
except ImportError:
    visvis_available = False
