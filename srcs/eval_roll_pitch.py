from sys import argv
import os
import errno
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import config
from helpers import mkdir_p

from plane_fitter import fit_plane

label_dir = config.data_dir + "labels/"
raw_dir = config.data_dir + "raw/"

if len(argv) != 2 or argv[1] not in ("static", "moving"):
    print("Usage: labeler.py {static, moving}")

dirs = [x for x in next(os.walk(raw_dir))[1]]
static = (argv[1] == "static")

if static:
    dirs = [x for x in dirs if x.startswith("stationary")]
else:
    dirs = [x for x in dirs if x.startswith("moving")]

rolls = []
pitches = []
for d in dirs:
    all_pts = []
    mkdir_p(os.path.join(label_dir, d))
    stationary = d.startswith("stationary")

    files = os.listdir(os.path.join(raw_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )

    for f in files:
        if not f.endswith(".npz"):
            continue

        f_path = os.path.join(raw_dir, d, f)
        data = np.load(f_path)
        pts = data['pointcloud']

        roll, pitch = fit_plane(pts)
        rolls.append(roll)
        pitches.append(pitch)

plt.subplot(1, 2, 1)
plt.hist(rolls)
plt.subplot(1, 2, 2)
plt.hist(pitches)
plt.show()
