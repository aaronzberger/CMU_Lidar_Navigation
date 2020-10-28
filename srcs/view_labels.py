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

from bev import  BEV #pointcloud_to_bev, visualize_bev, labels_to_bev, \
        #make_targets, decode_reg_map
from utils import load_config



config_f, _, _, _ = load_config("10meter")
geom = config_f["geometry"]
bev = BEV(geom)


label_dir = config.data_dir + "labels/"
raw_dir = config.data_dir + "raw/"


if len(argv) != 2 or argv[1] not in ("static", "moving"):
    print("Usage: labeler.py {static, moving}")

dirs = [x for x in next(os.walk(label_dir))[1]]
static = (argv[1] == "static")

if static:
    dirs = [x for x in dirs if x.startswith("stationary")]
else:
    dirs = [x for x in dirs if x.startswith("moving")]

for d in dirs:
    stationary = d.startswith("stationary")

    files = os.listdir(os.path.join(label_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )

    # print(files[0])
    # i = files.index("1574112100836612463.npz")
    # files = files[i:]

    shuffle(files)

    for f in files:
        if not f.endswith(".npz"):
            continue

        if stationary:
            label_path = os.path.join(label_dir, d, "all.npz") 
        else:
            label_path = os.path.join(label_dir, d, f)

        if os.path.exists(label_path):
            label_data = np.load(label_path)
            labels = bev.crop_labels(label_data['labels'])

            raw_path = os.path.join(raw_dir, d, f)
            raw_data = np.load(raw_path)
            pts = np.copy(raw_data['pointcloud'][:,0:3])

            # roll, pitch = fit_plane(plane_pts)

            # r = Rotation.from_euler('xy', [-roll, -pitch])
            # print(roll, pitch)
            # print(r.as_euler('xyz'))
            # R = r.as_matrix()

            roll = 0
            pitch = radians(-3.)

            R = euler_matrix(-roll, -pitch, 0)[0:3,0:3]
            pts = np.matmul(R, pts.T).T

            idx =  ((pts[:,0] > geom["W1"]) & 
                    (pts[:,0] < geom["W2"]) & 
                    (pts[:,1] < geom["L2"]) & 
                    (pts[:,1] > geom["L1"]))

            pts = pts[idx]

            zmin = geom["H1"]
            zmax = geom["H2"]

            # ref = np.copy(raw_data['pointcloud'][:,3])[idx]
            # pts_colors = np.zeros((len(ref), 3))
            #
            # for i in range(len(ref)):
            #     pts_colors[i,:] = ref[i] / 100.

            # pts_colors = cmap(pts_colors)
            # print(pts_colors)

            print("Labels:")
            print(labels)

            label_map, class_map, instance_map, num_instances = \
                    bev.make_targets(labels)

            fraction = float(np.count_nonzero(class_map)) / (class_map.shape[0] * class_map.shape[1])
            print("foreground fraction: %.6f" % fraction)

            bev_img = bev.pointcloud_to_bev(
                np.copy(raw_data['pointcloud'][:,0:3])
            )

            labels_bev = bev.labels_to_bev(labels)

            # bev.visualize_bev(bev_img, labels_bev, label_map, class_map,
            #        instance_map)

            # bev.decode_reg_map(label_map, class_map)
            
            bev.visualize_lines_3d(labels, pts, order='xxyy')


