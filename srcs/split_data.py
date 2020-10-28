from sys import argv
import os
import errno
from glob import glob

import csv

import config

label_dir = config.data_dir + "labels/"
raw_dir = config.data_dir + "raw/"
dataset_dir = os.path.join(config.data_dir, "dataset")

def write_split(splitname, files):
    with open(os.path.join(config.data_dir, splitname + ".csv"), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['raw', 'label'])
        writer.writeheader()

        for raw, label in files:
            writer.writerow({'raw': raw, 'label': label})

print(label_dir)
print(raw_dir)
dirs = [x for x in next(os.walk(label_dir))[1]]
filenames = []
for d in dirs:
    stationary = d.startswith("stationary")

    # Only moving for now
    if stationary:
        continue

    files = os.listdir(os.path.join(label_dir, d))
    files = sorted(list(files), key=lambda x: int(x[:-4]) )

    for f in files:
        if not f.endswith(".npz"):
            continue

        if stationary:
            label_path = os.path.join(label_dir, d, "all.npz") 
        else:
            label_path = os.path.join(label_dir, d, f)

        raw_path = os.path.join(raw_dir, d, f)
        if os.path.exists(label_path):
            filenames.append((raw_path, label_path))
        else:
            print("Warning: No labels found for raw example %s" % (raw_path))

# filenames = [n for n in range(20)]
n_files = len(filenames)

train_percent = 0.7
test_percent = .15
val_percent = .15

train_end = int(n_files * train_percent)
test_end = train_end + int(n_files * test_percent)

train_filenames = filenames[:train_end]
test_filenames = filenames[train_end:test_end]
val_filenames = filenames[test_end:]

# print(train_filenames, test_filenames, val_filenames)

write_split("train", train_filenames)
write_split("test", test_filenames)
write_split("val", val_filenames)
