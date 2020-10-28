import os
import errno
from math import sin, cos

import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rotation_2d(theta):
    return  np.array(
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]])
