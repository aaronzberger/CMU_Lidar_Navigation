from sys import argv
import os
import errno
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import config
from utils import mkdir_p
from keyboard_labeler import Labeler

# Where to save the labels (doesn't need to exist yet)
label_dir = "/home/aaron/Documents/velo_bags/labels"

# Where the .npz files with the point clouds are
# (If you don't have this yet, run bag_extractor.py)
raw_dir = "/home/aaron/Documents/velo_bags/bagsraw"

if len(argv) != 1:
    print("NOTE: No arguments expected for this file.")

# Gather the names of all directories of point cloud files
dirs = [x for x in next(os.walk(raw_dir))[1]]

labeler = Labeler()

for d in dirs:
    # Make a directory with the same name for the labels
    mkdir_p(os.path.join(label_dir, d))

    files = os.listdir(os.path.join(raw_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )

    for f in files:
        if not f.endswith(".npz"):
            continue

        label_path = os.path.join(label_dir, d, f)

        if os.path.exists(label_path):
            print("The labels for %s already exist! Delete the file and re-run this to replace them" % os.path.join(d, f))
            continue

        file_path = os.path.join(raw_dir, d, f)
        data = np.load(file_path)
        pts = data['pointcloud']

        labeler.plot_points(pts)             
        np.savez(label_path, labels=labeler.get_labels())