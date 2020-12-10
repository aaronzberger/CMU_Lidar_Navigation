from sys import argv
import os
import errno
from glob import glob

import rosbag

import numpy as np
from matplotlib import pyplot as plt

from numpy_pc2 import pointcloud2_to_xyzi_array
import config
from utils import mkdir_p
from keyboard_labeler import Labeler

label_dir = config.data_dir + 'labels/'
raw_dir = config.data_dir + 'raw/'

if len(argv) != 2 or argv[1] not in ('static', 'moving'):
    print 'Usage: labeler.py {static, moving}'

dirs = [x for x in next(os.walk(raw_dir))[1]]
static = (argv[1] == 'static')

if static:
    dirs = [x for x in dirs if x.startswith('stationary')]
else:
    dirs = [x for x in dirs if x.startswith('moving')]

labeler = Labeler()
for d in dirs:
    all_pts = []
    mkdir_p(os.path.join(label_dir, d))
    stationary = d.startswith('stationary')

    files = os.listdir(os.path.join(raw_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )

    print files

    for f in files:
        if not f.endswith('.npz'):
            continue

        if stationary:
            label_path = os.path.join(label_dir, d, 'all.npz') 
        else:
            label_path = os.path.join(label_dir, d, f)

        if os.path.exists(label_path):
            print 'path %s exists!' % os.path.join(d, f)
            continue

        f_path = os.path.join(raw_dir, d, f)
        data = np.load(f_path)
        pts = data['pointcloud']

        if stationary:
            all_pts.append(pts)
        else:
            labeler.plot_points(pts)             
            labeler.get_labels()

    if stationary and len(all_pts) != 0:
        pts = np.concatenate(all_pts, axis=0)
        labeler.plot_points(pts)
        labeler.get_labels()
