from math import radians

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from transformations import euler_matrix
import config

from visualization import pointcloud, horizontal_lines, vertical_lines
import matplotlib
import open3d as o3d


class BEV:
    def __init__(self, geom):
        self.width, self.height, self.channels = geom["input_shape"]
        self.mins = np.array([geom[s] for s in ("W1", "L1", "H1")])
        self.maxes = np.array([geom[s] for s in ("W2", "L2", "H2")])

        self.resolutions = np.array([
            abs(geom["W2"] - geom["W1"]) / self.width,
            abs(geom["L2"] - geom["L1"]) / self.height,
            abs(geom["H2"] - geom["H1"]) / self.channels,
        ])

        self.geom = geom

        self.output_downsample_fac = (
            int(geom["input_shape"][0]) //
            int(geom["label_shape"][0]))

        self.output_width = self.width // self.output_downsample_fac
        self.output_height = self.height // self.output_downsample_fac

        # For cropping the labels
        # Array of lines in homogeneous coordinates
        # All normals point inwards, dot product of in-bounds coordinates and
        # bounding line should be positive
        self.bounds = np.array([
            [1, 0,   -geom["W1"]],
            [-1, 0,  geom["W2"]],
            [0, 1,   -geom["L1"]],
            [0, -1,  geom["L2"]]
        ])

        self.cnorm_orig = matplotlib.colors.Normalize(vmin=geom["H1"],
                vmax=geom["H2"])

        print(self.resolutions)
        self.cnorm_bev = matplotlib.colors.Normalize(
            vmin=(geom["H1"] - self.mins[2]) / self.resolutions[0],
            vmax=(geom["H2"] - self.mins[2]) / self.resolutions[0])

    cmap_jet = matplotlib.pyplot.cm.get_cmap('jet')

    # Decode angle encoded as sin(2*theta), cos(2*theta)
    @classmethod
    def decode_angle(cls, sin_2t, cos_2t):
      sin_t = np.sqrt(.5*(1.-cos_2t))
      cos_t = sin_2t / (2. * sin_t)
      return sin_t, cos_t, np.arctan2(sin_t, cos_t)

    def crop_labels(self, labels):
        N = len(labels)

        if N == 0:
            print("No rows found in this point cloud")
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
        # Counter for tilt of sensor
        pitch = radians(3.)
        R_extrinsic = euler_matrix(0, pitch, 0)[0:3,0:3]

        pts = np.matmul(R_extrinsic, pts.T).T


        # Crop the pointcloud
        pts = pts[
                (pts[:,0] > self.geom["W1"]) & 
                (pts[:,0] < self.geom["W2"]) & 
                (pts[:,1] > self.geom["L1"]) &
                (pts[:,1] < self.geom["L2"]) & 
                (pts[:,2] > self.geom["H1"]) & 
                (pts[:,2] < self.geom["H2"])]

        return pts
            

    def pointcloud_to_bev(self, pts):
        bev = np.zeros((self.width, self.height, self.channels), dtype='float32')

        # Random rotation augmentation
        H = np.eye(3)
        # H[1,1] = -1

        pts = self.standardize_pointcloud(pts)

        pts -= self.mins
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
        # Translate into BEV frame
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

    def decode_reg_map(self, reg_map, class_map, threshold):
        xs, ys = np.meshgrid(
            np.arange(self.output_width),
            np.arange(self.output_height),
            indexing='ij'
        )

        coords = np.concatenate((
            xs.reshape(-1, 1),
            ys.reshape(-1, 1)
        ), axis=1).astype('int')

        # print(reg_map.shape, class_map.shape)
        reg_map = reg_map.reshape(-1, reg_map.shape[2])
        class_map = class_map.reshape(-1)

        # print(reg_map.shape, class_map.shape)

        mask = class_map > threshold

        lines = reg_map[mask, :]
        coords = coords[mask, :]

        n_target = lines[:,0:2]
        d_target = lines[:,2]

        d = -(d_target - np.sum(coords * n_target, axis=1))
        n = n_target / np.linalg.norm(n_target, axis=1).reshape(-1, 1)

        lines_decoded = np.concatenate(
            (n, d.reshape(-1, 1)), 
            axis=1
        )

        scores = class_map[mask]

        # unique_lines = np.unique(lines_decoded, axis=0)
        # print(unique_lines)
        # print(unique_lines.shape)
        

        return lines_decoded, scores

    @classmethod
    def make_bev_viz_image(cls, bev):
        values = bev.argmax(axis=2).astype('float')
        values = (values / float(bev.shape[2])).reshape(-1)
        img_shape = (bev.shape[0], bev.shape[1], 4)
        img = BEV.cmap_jet(values).reshape(img_shape)
        img[np.bitwise_not(np.any(bev, axis=2))] = 0.

        return img

    def visualize_lines_3d(self, lines, pts, frame='bev', order='xyxy'):
        if order != 'xxyy' and order != 'xyxy':
            raise ValueError("order must be one of {xxyy, xyxy}")

        if frame != 'bev' and frame != 'orig':
            raise ValueError("frame must be one of {frame, orig}")

        if frame == "bev":
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

        if frame == "bev":
            zmin = (zmin - self.mins[2]) / self.resolutions[0]
            zmax = (zmax - self.mins[2]) / self.resolutions[0]

        o3d.visualization.draw_geometries([
            pointcloud(pts, colors=pts_colors), 
            horizontal_lines(lines1, lines2, zmin),
            horizontal_lines(lines1, lines2, zmax),
            vertical_lines(lines1, zmin, zmax),
            vertical_lines(lines2, zmin, zmax),
        ])

    def visualize_bev(self, bev, labels, reg_map, class_map,
        instance_map=None, cls_preds=None, geom=None):
        img = BEV.make_bev_viz_image(bev)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img)

        plt.xlim((0, self.width))
        plt.ylim((self.height, 0))

        for l in labels:
            plt.plot(l[2:4], l[0:2], c='r', linewidth=.5)


        if cls_preds is not None:
          plt.subplot(1, 3, 2)
          plt.axis('off')
          plt.tight_layout()
          plt.imshow(cls_preds, cmap='jet')

        # plt.subplot(1, 2, 2)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.imshow(class_map)

        shuffle = np.zeros(256)
        shuffle[1:] = np.random.permutation(255)


        cmap = cm.get_cmap('gist_rainbow')
        #print(instance_map.shape)
        #print(class_map.shape)

        if instance_map is not None:
          instance_map *= np.arange(32).reshape(-1, 1, 1)
          instance_map = np.sum(instance_map, axis=0).astype('int') / len(labels)

          plt.subplot(1, 2, 2)
          plt.axis('off')
          plt.tight_layout()
          values = (instance_map.astype('float32')).reshape(-1)

          colormap_shape = (instance_map.shape[0], instance_map.shape[1], 4)
          colormap = cmap(values).reshape(colormap_shape)
          colormap[reg_map[:,:,0] == 0] = 0.

          plt.imshow(colormap)

        # plt.subplot(2, 2, 3)
        # plt.axis('off')
        # plt.tight_layout()
        #
        # xs, ys = np.meshgrid(
        #     np.arange(self.output_width),
        #     np.arange(self.output_height),
        #     indexing='ij'
        # )
        #
        # coords = np.concatenate((
        #     xs.reshape(-1, 1),
        #     ys.reshape(-1, 1)
        # ), axis=1).astype('int')
        #
        # mask = class_map.reshape(-1).astype('bool')
        # decoded = reg_map.reshape(-1, reg_map.shape[2])[mask]
        #
        # p0 = coords[mask]
        # p1 = p0 + decoded[:,2:4]
        #
        # # sign = decoded[:, 2]
        # encoded_angles = decoded[:,0:2]
        #
        # for (x0, y0), (x1, y1), (sin_2t, cos_2t) in zip(p0, p1, encoded_angles):
        #     # Plot normal
        #     # if sign > 0:
        #     plt.plot((y0, y1), (x0, x1), c='b')
        #
        #
        #     _, _, t = BEV.decode_angle(sin_2t, cos_2t)
        #     #print("angles", np.arctan2(y1 - y0, x1 - x0), t)
        #
        #     dx1 = x0 + np.cos(t + np.pi/2)
        #     dx2 = x0 + np.cos(t - np.pi/2)
        #
        #     dy1 = y0 + np.sin(t + np.pi/2)
        #     dy2 = y0 + np.sin(t - np.pi/2)
        #
        #     plt.plot((dy1, dy2), (dx1, dx2), c='r')
        #
        #     # elif sign < 0:
        #     #   plt.plot((y0, y1), (x0, x1), c='r')
        #
        #     # Plot line
        #     # plt.plot(
        #     #     (y1 + (d_y * 200), y1 - (d_y * 200)), 
        #     #     (x1 + (d_x * 200), x1 - (d_x * 200)), 
        #     #     c='g'
        #     # )
        #
        # # plot label GT
        # for l in labels:
        #     plt.plot(l[2:4] / 4, l[0:2] / 4, c='g', linewidth=.5)
        #
        # plt.imshow(class_map)
        #
        plt.show()
