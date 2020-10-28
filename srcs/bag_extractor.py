from sys import argv
import os
import errno
from glob import glob

import rosbag
import numpy as np

import config

from numpy_pc2 import pointcloud2_to_xyzi_array
from helpers import mkdir_p

pointcloud_topic = "/velodyne_points"

raw_dir = config.data_dir + "raw/"
bags_dir = config.data_dir + "bags/"

ext = '.bag'

bagnames = [y for x in os.walk(bags_dir) for y in glob(os.path.join(x[0], '*' + ext))]

filenames = [b[len(bags_dir):-len(ext)].replace("/","_") for b in bagnames]


for filename, bagname in zip(filenames, bagnames):
    bag = rosbag.Bag(bagname)
    data = bag.read_messages(
        topics=[pointcloud_topic])

    count = 0
    for topic, msg, t in data:
        if topic == pointcloud_topic:
            pc = pointcloud2_to_xyzi_array(msg) 
            timestamp = msg.header.stamp.to_nsec()
            timestamp_arr = np.array([timestamp], dtype='int64')
            
            d = "%s%s" % (raw_dir, filename)
            mkdir_p(d)

            path =  "%s/%s" % (d, timestamp)

            np.savez(path, pointcloud=pc, timestamp=timestamp_arr)
