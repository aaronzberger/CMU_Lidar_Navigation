from sys import argv
import argparse
import os
import errno
from glob import glob

import csv
from config import exp_name
from utils import load_config

config, _, _, _ = load_config(exp_name)

# Directory containing the labels generated from run_labeler.py
label_dir = os.path.join(config['data_dir'], 'labels')

# Directory containing the the .npz files generated by bag_extractor.py
raw_dir = os.path.join(config['data_dir'], 'raw')

def write_split(splitname, files):
    '''
    Write a csv file with paths to the raw data and labeled data

    Parameters:
        splitname (string): the name of the portion of the data (train, test, val, etc)
        files (arr): list of tuples of raw path and label path
    '''
    with open(os.path.join(config['data_dir'], splitname + '.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['raw', 'label'])
        writer.writeheader()

        for raw, label in files:
            writer.writerow({'raw': raw, 'label': label})

print('Gathering labels from %s' % label_dir)
print('Gathering raw data from %s' % raw_dir)

# Find the names of all directories in label_dir
dirs = [x for x in next(os.walk(label_dir))[1]]

# Store tuples of the paths to the raw and labeled data
filenames = []

for d in dirs:
    files = os.listdir(os.path.join(label_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )

    for f in files:
        if not f.endswith('.npz'):
            continue

        label_path = os.path.join(label_dir, d, f)

        # The path to the raw data should be the same (see run_labeler.py)
        raw_path = os.path.join(raw_dir, d, f)

        if os.path.exists(label_path):
            filenames.append((raw_path, label_path))
        else:
            print('No labels found for %s. Go run run_labeler.py to finish labeling.' % raw_path)

n_files = len(filenames)

train_percent = 0.9
test_percent = 0.1

train_end = int(n_files * train_percent)

train_filenames = filenames[:train_end]
test_filenames = filenames[train_end:]

write_split('train', train_filenames)
write_split('test', test_filenames)

print('''
    Data split:

        Train:       {}%    {} point clouds
        Test:        {}%    {} point clouds
'''.format(train_percent * 100, train_end, test_percent * 100, n_files - train_end))