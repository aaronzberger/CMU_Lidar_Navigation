'''
This file is not run independently

Helper functions for the BEV representation, where X and Y correspond to width and height
'''
from math import radians

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from transformations import euler_matrix

import matplotlib
import open3d as o3d


class BEV:
    def __init__(self, geom):
        self.width, self.height, self.channels = geom['input_shape']
        self.mins = np.array([geom[s] for s in ('W1', 'L1', 'H1')])
        self.maxes = np.array([geom[s] for s in ('W2', 'L2', 'H2')])

        self.resolutions = np.array([
            abs(geom['W2'] - geom['W1']) / self.width,
            abs(geom['L2'] - geom['L1']) / self.height,
            abs(geom['H2'] - geom['H1']) / self.channels,
        ])

        self.geom = geom

        self.output_downsample_fac = (
            int(geom['input_shape'][0]) //
            int(geom['label_shape'][0]))

        self.output_width = self.width // self.output_downsample_fac
        self.output_height = self.height // self.output_downsample_fac

        # For cropping the labels
        # Array of lines in homogeneous coordinates
        # All normals point inwards, dot product of in-bounds coordinates and
        # bounding line should be positive
        self.bounds = np.array([
            [1, 0,   -geom['W1']],
            [-1, 0,  geom['W2']],
            [0, 1,   -geom['L1']],
            [0, -1,  geom['L2']]
        ])

        self.cnorm_orig = matplotlib.colors.Normalize(vmin=geom['H1'],
                vmax=geom['H2'])
    
        self.cnorm_bev = matplotlib.colors.Normalize(
            vmin=(geom['H1'] - self.mins[2]) / self.resolutions[2],
            vmax=(geom['H2'] - self.mins[2]) / self.resolutions[2])

    cmap_jet = matplotlib.pyplot.cm.get_cmap('jet')

    def crop_labels(self, labels):
        '''
        Edit a list of lines so all points are in-bounds and lines are valid

        Parameters:
            labels (arr): the list of labels

        Returns:
            arr: a new, edited, list of labels
        '''
        N = len(labels)

        if N == 0:
            print('No rows found in this point cloud')
            return labels

        # Homogeneous coordinates of point1 and point2 for the lines
        p1 = np.ones((N, 3))
        p1[:,0:2] = labels[:,[0, 2]]

        p2 = np.ones((N, 3))
        p2[:,0:2] = labels[:,[1, 3]]

        # Cross product of two points gives the line
        lines = np.cross(p1, p2)
        lines /= np.linalg.norm(lines[:,0:2])

        # Iteratively crop the lines with each bounding line
        for bound in self.bounds:
            # Compute which points are in-bounds
            N = len(lines)
            dists = np.zeros((2, N))
            valid = np.zeros((2, N), dtype='bool')
            
            for i, p in enumerate((p1, p2)):
                bound /= np.linalg.norm(bound)
                dists[i] = np.sum(p * bound, axis=1)
                valid[i] = dists[i] > 0
            
            # Discard line segments which are totally out of bounds
            # i.e. cannot be cropped in order to make in-bounds
            either_valid = np.any(valid, axis=0)

            p1 = p1[either_valid]
            p2 = p2[either_valid]
            lines = lines[either_valid]
            valid = valid.T[either_valid].T

            # Crop line segments to make them in-bounds
            for i, p in enumerate((p1, p2)):
                idxs = np.where(np.bitwise_not(valid[i]))[0]
                if len(idxs > 0):
                    # Compute intersection of lines and boundary
                    p_new = np.cross(lines[idxs], bound)
                    p_new /= p_new[:,2].reshape(-1, 1)

                    # Replace old points with new intersected points
                    p[idxs] = p_new

        labels_new = np.zeros((lines.shape[0], 4))
        labels_new[:,[0, 2]] = p1[:,0:2]
        labels_new[:,[1, 3]] = p2[:,0:2]

        return labels_new

    def standardize_pointcloud(self, pts):
        '''
        Make the point cloud valid for viewing

        Parameters:
            pts (arr): the point cloud

        Returns:
            arr: a corrected point cloud
        '''
        # Correct for the tilt of the sensor
        pitch = radians(3.)
        R_extrinsic = euler_matrix(0, pitch, 0)[0:3,0:3]

        pts = np.matmul(R_extrinsic, pts.T).T

        # Crop the pointcloud
        pts = pts[
                (pts[:,0] > self.geom['W1']) & 
                (pts[:,0] < self.geom['W2']) & 
                (pts[:,1] > self.geom['L1']) &
                (pts[:,1] < self.geom['L2']) & 
                (pts[:,2] > self.geom['H1']) & 
                (pts[:,2] < self.geom['H2'])]

        return pts
            
    def pointcloud_to_bev(self, pts):
        '''
        Convert a raw point array to a bev-represented array

        Parameters:
            pts (arr): raw point array [X, Y, Z]
        
        Returns:
            numpy.ndarray: [width, height, channels] array of the point cloud
        '''
        bev = np.zeros((self.width, self.height, self.channels), dtype='float32')

        pts = self.standardize_pointcloud(pts)

        pts -= self.mins

        # Scale the points from robot coordinates to image pixels
        pts /= self.resolutions

        coords = pts.astype('int')

        bev[coords[:,0], coords[:,1], coords[:,2]] = 1.
        
        return bev

    # source https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
    def _naive_line(self, r0, c0, r1, c1):
        # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
        # If either of these cases are violated, do some switches.
        if abs(c1-c0) < abs(r1-r0):
            # Switch x and y, and switch again when returning.
            xx, yy = self._naive_line(c0, r0, c1, r1)
            return (yy, xx)

        # At this point we know that the distance in columns (x) is greater
        # than that in rows (y). Possibly one more switch if c0 > c1.
        if c0 > c1:
            return self._naive_line(r1, c1, r0, c0)

        # We write y as a function of x, because the slope is always <= 1
        # (in absolute value)
        x = np.arange(c0, c1+1, dtype=float)
        y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

        valbot = np.floor(y)-y+1
        valtop = y-np.floor(y)

        return (np.floor(y), x)

        # return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int),
        #         np.concatenate((valbot, valtop)))

    def labels_to_bev(self, labels):
        '''
        Convert the given lines in two-points format to the correct pixel space

        Parameters:
            labels (arr): the labels
        
        Returns:
            arr: the labels, transformed into bev using given geometry
        '''
        if len(labels) == 0:
          return labels
        else:
          labels = (labels - self.mins[[0, 0, 1, 1]]) /\
                  self.resolutions[[0, 0, 1, 1]]

          return labels

    max_num_instances = 32
    def make_targets(self, labels):
        labels = self.labels_to_bev(labels) / self.output_downsample_fac

        reg_map = np.zeros(                                                     
            (self.output_width, self.output_height, self.reg_map_channels), 
            dtype='float32')

        class_map = np.zeros(
            (self.output_width, self.output_height), 
            dtype='float32')

        instance_map = np.zeros((
          self.max_num_instances,
          self.output_width, 
          self.output_height), 
            dtype='float32')

        n_ins = 0
        for i, (x0, x1, y0, y1) in enumerate(labels):
            xs, ys = self._naive_line(x0, y0, x1, y1)

            valid = (
                (xs > 0) & 
                (xs < self.output_width) & 
                (ys < self.output_height) & 
                (ys > 0))
            xs = xs[valid].astype('int')
            ys = ys[valid].astype('int')

            if len(xs) > 0 and len(ys) > 0:
                class_map[xs, ys] = 1.
                instance_map[n_ins, xs, ys] = 1.

                n_ins += 1

        n_instances = n_ins

        return reg_map, class_map, instance_map, n_instances

    reg_map_channels = 4

    def pointcloud(self, points, colors=None):
        '''Helper function for the method visualize_lines_3d'''
        pcl = o3d.geometry.PointCloud()

        if colors is not None:
            colors = colors.astype('float64')
            pcl.colors = o3d.utility.Vector3dVector(colors)

        pcl.points = o3d.utility.Vector3dVector(points.astype('float64'))

        return pcl

    def visualize_lines_3d(self, lines, pts, frame='bev', order='xyxy'):
        '''
        Use Open3D to plot labels on 3D point clouds

        Parameters:
            lines (arr): the labels
            pts (arr): the bev-represented point cloud
            frame (string): the representation to use
            order (string): the order the coordinates are in in the lines parameter
        '''
        if order != 'xxyy' and order != 'xyxy':
            raise ValueError('order must be one of {xxyy, xyxy}')

        if frame != 'bev' and frame != 'orig':
            raise ValueError('frame must be one of {frame, orig}')

        if frame == 'bev':
            pts_colors = self.cmap_jet(self.cnorm_bev(pts[:,2]))[:,0:3]
        else:
            pts_colors = self.cmap_jet(self.cnorm_orig(pts[:,2]))[:,0:3]

        if len(lines) > 0:
            if order == 'xxyy':
                lines1 = lines[:,[0,2]]
                lines2 = lines[:,[1,3]]
            else:
                lines1 = lines[:,0:2]
                lines2 = lines[:,2:4]
        else:
            lines1 = np.zeros((0, 2))
            lines2 = np.zeros((0, 2))

        zmin = self.mins[2]
        zmax = self.maxes[2]

        if frame == 'bev':
            zmin = (zmin - self.mins[2]) / self.resolutions[0]
            zmax = (zmax - self.mins[2]) / self.resolutions[0]

        # Make Open3D lines
        zeros = np.full((lines1.shape[0], 1), zmin)
        pts1 = np.concatenate((lines1, zeros), axis=1)
        pts2 = np.concatenate((lines2, zeros), axis=1)

        corresp = [(n, n) for n in range(len(pts1))]

        o3d_lines =\
        o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                self.pointcloud(pts1),
                self.pointcloud(pts2), 
                corresp
        )

        o3d.visualization.draw_geometries([
            self.pointcloud(pts, colors=pts_colors), o3d_lines
        ])