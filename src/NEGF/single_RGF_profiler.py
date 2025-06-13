import numpy as np
import multiprocessing
from itertools import product
import time

# Assume these classes are defined in their respective files
from device import Device
from rgf import GreensFunction

device = Device()
rgf = GreensFunction(device)
G_R = rgf.sparse_rgf_G_R(0.1, 0.1)