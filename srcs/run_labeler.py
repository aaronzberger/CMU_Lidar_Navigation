from sys import argv
import os
import errno
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from config import exp_name
from utils import mkdir_p, load_config
from keyboard_labeler import Labeler
from tqdm import tqdm

config, _, _, _ = load_config(exp_name)

# Where to save the labels (doesn't need to exist yet)
label_dir = os.path.join(config['data_dir'], 'labels')

# Where the .npz files with the point clouds are
# (If you don't have this yet, run bag_extractor.py)
raw_dir = os.path.join(config['data_dir'], 'raw')

if len(argv) != 1:
    print('NOTE: No arguments expected for this file.')

# Gather the names of all directories of point cloud files
dirs = [x for x in next(os.walk(raw_dir))[1]]

# Count the number of raw files (regardless of whether they have already been labeled)
num_raw_files = 0
for d in dirs:
    num_raw_files += len([name for name in os.listdir(os.path.join(raw_dir, d)) if name.endswith('.npz')])

num_labels_already = 0
for d in [y for y in next(os.walk(label_dir))[1]]:
    num_labels_already += len([name for name in os.listdir(os.path.join(label_dir, d)) if name.endswith('.npz')])

print('Found labels for %s/%s raw files. %s to go!' % (num_labels_already, num_raw_files, num_raw_files - num_labels_already))

labeler = Labeler()

# Make a progress bar for labeling
with tqdm(total=num_raw_files, desc='Labeling: ', unit='pcl', initial=num_labels_already) as progress:
    for d in dirs:
        # Make a directory with the same name for the labels
        mkdir_p(os.path.join(label_dir, d))

        files = os.listdir(os.path.join(raw_dir, d))
        files = sorted(list(files), key=lambda x: int(x[:-4]) )

        for f in files:
            if not f.endswith('.npz'):
                continue

            label_path = os.path.join(label_dir, d, f)

            if os.path.exists(label_path):
                continue

            file_path = os.path.join(raw_dir, d, f)
            data = np.load(file_path)
            pts = data['pointcloud']

            labeler.plot_points(pts)             
            np.savez(label_path, labels=labeler.get_labels())
            progress.update()