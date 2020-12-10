'''
Usage: bag_extractor.py

Extracts point clouds from bag files in a directory and saves them 
in the correct format and location
'''

from sys import argv
import os
import errno
from glob import glob

import rosbag
import numpy as np

from config import exp_name

from numpy_pc2 import pointcloud2_to_xyzi_array
from utils import mkdir_p, load_config

pointcloud_topic = '/velodyne_points'
ext = '.bag'

# Directory to save the .npz files containing point clouds
config, _, _, _ = load_config(exp_name)
save_dir = os.path.join(config['data_dir'], 'raw')

# Directory containing the original bag files
bags_dir = '/home/aaron/Documents/velo_bags/bags/'

# Collect the names of all the bags in the directory specified above
bagnames = [y for x in os.walk(bags_dir) for y in glob(os.path.join(x[0], '*' + ext))]
filenames = [b[len(bags_dir):-len(ext)].replace('/','_') for b in bagnames]

num_pcl = 0

for filename, bagname in zip(filenames, bagnames):
    bag = rosbag.Bag(bagname)
    data = bag.read_messages(
        topics=[pointcloud_topic])

    for topic, msg, t in data:
        if topic == pointcloud_topic:
            # Convert pcl to array for saving
            pc = pointcloud2_to_xyzi_array(msg) 

            timestamp = msg.header.stamp.to_nsec()
            timestamp_arr = np.array([timestamp], dtype='int64')
            
            # Create a directory named the same as the bag in which to save the data
            pcl_dir = os.path.join(save_dir, filename)
            mkdir_p(pcl_dir)

            # Save the point cloud with the name as the timestamp of this message
            np.savez(os.path.join(pcl_dir, str(timestamp)), pointcloud=pc, timestamp=timestamp_arr)
            num_pcl += 1

print('''\n
    Finished extracting point clouds:
    
        Bags:           {}
        Point Clouds:   {}
\n'''.format(len(bagnames), num_pcl))
