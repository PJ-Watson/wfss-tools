"""
Grism-related codes.
"""

# from wfss_tools.grism.fitting_tools import *
# from wfss_tools.grism.multiregion import *
# from wfss_tools.grism.specgen import *
# from wfss_tools.grism.utils import *

import os

import numpy as np

# Allow for environment variable override if necessary
float_dtype = np.dtype(os.getenv("MULTIREGION_FLOAT_DTYPE", "f8")).type
