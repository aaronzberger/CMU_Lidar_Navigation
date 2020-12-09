from sys import argv
import os
import errno
from glob import glob

import rosbag
import numpy as np

import config

from numpy_pc2 import pointcloud2_to_xyzi_array
from utils import mkdir_p

pointcloud_topic = "/velodyne_points"
ext = '.bag'

# Directory where .npz files containing the point clouds will be saved
save_dir ="/home/aaron/Documents/velo_bags/bagsraw/"

# Directory containing the original bag files
bags_dir = "/home/aaron/Documents/velo_bags/bags/"

# Collect the names of all the bags in the directory specified above
bagnames = [y for x in os.walk(bags_dir) for y in glob(os.path.join(x[0], '*' + ext))]
filenames = [b[len(bags_dir):-len(ext)].replace("/","_") for b in bagnames]

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
            pcl_dir = "%s%s" % (save_dir, filename)
            mkdir_p(pcl_dir)

            # Save the point cloud with the name as the timestamp of this message
            path =  "%s/%s" % (pcl_dir, timestamp)
            np.savez(path, pointcloud=pc, timestamp=timestamp_arr)
