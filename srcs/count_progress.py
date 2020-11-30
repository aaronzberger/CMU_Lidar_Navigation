from sys import argv
import os
import errno
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import config
from helpers import mkdir_p

label_dir = config.data_dir + "labels/"
raw_dir = config.data_dir + "raw/"

dirs = [x for x in next(os.walk(raw_dir))[1]]

total_static = 0
total_moving = 0
labeled_moving = 0
labeled_static = 0
total = 0
labeled = 0

for d in dirs:
    stationary = d.startswith("stationary")

    if stationary:
        total += 1
        total_static += 1

        label_path = os.path.join(label_dir, d, "all.npz") 
        if os.path.exists(label_path):
            labeled_static += 1
            labeled += 1

    for f in os.listdir(os.path.join(raw_dir, d)):
        if not f.endswith(".npz"):
            continue

        if not stationary:
            label_path = os.path.join(label_dir, d, f)
            total += 1
            total_moving += 1

            if os.path.exists(label_path):
                labeled_moving += 1
                labeled += 1

print("total moving: %d" % total_moving)
print("total labeled: %d" % labeled_moving)
print("progress %.3f" % (float(labeled_moving) / total_moving))
print("\n")
print("total static: %d" % total_static)
print("total labeled: %d" % labeled_static)
print("progress %.3f" % (float(labeled_static) / total_static))
print("\n")
print("total frames: %d" % total)
print("total labeled: %d" % labeled)
print("progress %.3f" % (float(labeled) / total))
