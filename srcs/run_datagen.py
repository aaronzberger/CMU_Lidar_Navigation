from sys import argv
import os
import errno
from glob import glob
from scipy.spatial.transform import Rotation
from random import shuffle

import matplotlib
from matplotlib import pyplot as plt

from transformations import euler_matrix

import numpy as np
import open3d as o3d

import config

from plane_fitter import fit_plane
from math import radians
from utils import mkdir_p

from bev import BEV # pointcloud_to_bev, visualize_bev, labels_to_bev

label_dir = config.data_dir + "labels/"
raw_dir = config.data_dir + "raw/"
dataset_dir = os.path.join(config.data_dir, "dataset")

dirs = [x for x in next(os.walk(label_dir))[1]]

for d in dirs:
    stationary = d.startswith("stationary")

    files = os.listdir(os.path.join(label_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )
    # shuffle(files)

    for f in files:
        if not f.endswith(".npz"):
            continue

        if stationary:
            label_path = os.path.join(label_dir, d, "all.npz") 
        else:
            label_path = os.path.join(label_dir, d, f)

        raw_path = os.path.join(raw_dir, d, f)
        if os.path.exists(label_path):
            label_data = np.load(label_path)

            raw_data = np.load(raw_path)
            pts = np.copy(raw_data['pointcloud'][:,0:3])

            lines1 = label_data['labels'][:,[0,2]]
            lines2 = label_data['labels'][:,[1,3]]

            bev = pointcloud_to_bev(
                np.copy(raw_data['pointcloud'][:,0:3])
            )
            labels = labels_to_bev(label_data['labels'])
            # visualize_bev(bev, labels)
            
            dataset_path = os.path.join(dataset_dir, "%s_%s" % (d, f))
            np.savez(dataset_path, bev=bev, labels=labels)
        else:
            print("Warning: No labels found for raw example %s" % (raw_path))
