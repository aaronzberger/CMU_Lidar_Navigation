import rosbag

from math import pi
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_matrix, euler_from_quaternion, \
                               quaternion_from_euler, quaternion_multiply

from tf2_sensor_msgs import do_transform_cloud
from numpy_pc2 import pointcloud2_to_xyz_array, array_to_xyz_pointcloud2f
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from std_msgs.msg import Header

from mpl_toolkits.mplot3d import Axes3D
from sys import argv

class BagProcessor(object):
    def __init__(self, pos_offset=None, proc_pc=True):

        self.gps_topic = '/odometry/utm'
        self.pc_topic = '/quanergySensor/quanergyPoints'
        self.imu_topic = '/mti/sensor/imu'
        self.encoders_topic = '/wheel_encoder/odom'

        self.drive_cmd_topic = '/base/cmd_vel'

        self.proc_pc = proc_pc

        self.pos_offset = pos_offset

        self.drive_cmd = None

        # bagname = 'data/rivendale-row_02_e2w_2018-10-05-12-11-26.bag'
        # bagname = '/home/john/mnt/navigation_data/rivendale-small_row_03_e2w_2018-10-05-13-33-01.bag'
        # bagname = '/home/john/mnt/navigation_data/' + argv[1]

        h = Header(frame_id='base_link')
        q = Quaternion(0.,0.25881905,0.,0.96592583)
        xyz = Vector3(0.14145, 0.0, 1.5309) 
        t = Transform(rotation=q, translation=xyz)
        self.base_to_quan = TransformStamped(child_frame_id='quanergySensor', header=h, transform=t)
        self.map_to_base = TransformStamped()
        self.map_to_base_valid = False

        self.time = None

    def gps_cb(self, msg):
        pass

    def imu_cb(self, msg):
        pass

    def pointcloud_cb(self, msg, pts_base, pts_map):
        pass

    def encoders_cb(self, msg):
        pass

    def done_proc_cb(self):
        pass

    def process(self, bagname):
        bag = rosbag.Bag(bagname)
        data = bag.read_messages(
                topics=[self.gps_topic, self.pc_topic, self.imu_topic,
                    self.encoders_topic, self.drive_cmd_topic])
        
        got_imu_msg = False
        got_gps_msg = False
        curr_pos = None

        # offset = np.load('offset.npy')
        yaw_offset = None
        points_arr = []
        count = 0
        for topic, msg, t in data:
            self.time = t

            if topic == self.drive_cmd_topic:
                self.drive_cmd = msg

            if topic == self.gps_topic:
                got_gps_msg = True
                p = msg.pose.pose.position
                
                if self.pos_offset is None:
                    self.pos_offset = np.array([p.x, p.y])
                    curr_pos = np.zeros(2)
                else: 
                    curr_pos[:] = (p.x, p.y)
                    curr_pos -= self.pos_offset
                
                self.map_to_base.transform.translation.x = curr_pos[0]
                self.map_to_base.transform.translation.y = curr_pos[1]

                self.gps_cb(msg)

            if topic == self.imu_topic:
                got_imu_msg = True
                self.map_to_base.transform.rotation = msg.orientation
                self.imu_cb(msg)

            if topic == self.pc_topic and got_imu_msg and got_gps_msg:
                count += 1

                pts_base = None
                pts_map = None
                    
                if self.proc_pc:
                    print('Processing pointcloud number', count)
                    pc_base = do_transform_cloud(msg, self.base_to_quan)

                    pts_base = pointcloud2_to_xyz_array(pc_base)
                    test = np.bitwise_and(pts_base[:,1] < 1.5, pts_base[:,1] > -1.5)

                    pts_base = pts_base[test]
                    pc_base = array_to_xyz_pointcloud2f(pts_base)
                    pc_map = do_transform_cloud(pc_base, self.map_to_base)

                    pts_map = pointcloud2_to_xyz_array(pc_map)
                
                self.pointcloud_cb(msg, pts_base, pts_map)

            if topic == self.encoders_topic:
                self.encoders_cb(msg)

        self.done_proc_cb()
        bag.close()

