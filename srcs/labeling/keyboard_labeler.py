'''
This file is not run independently

Matplotlib tool for labeling lines on point clouds. Run from run_labeler.py
'''

from math import radians
import sys
import os

from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import rotation_2d, load_config
from config import exp_name

# Adapted from
# https://stackoverflow.com/questions/33569626/matplotlib-responding-to-click-events
# https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing


class Labeler(object):
    def __init__(self, view_only=False):
        self.view_only = view_only

        self.config, _, _, _ = load_config(exp_name)

        # Distance between rows
        self.row_spacing = 3.0

        # Length of rows
        self.row_radius = 20

        # format is x0, y0, theta
        self.row_to_sensor_t = np.array([0., 0.])
        self.row_to_sensor_angle = 0

        # Represents the guiding lines (X1, Y1, X2, Y2)
        self.rows = np.array([
            [-self.row_radius, n*self.row_spacing/2.,
             self.row_radius, n*self.row_spacing/2.]
            for n in range(-7, 7+2, 2)
        ])
        self.lines = []

    def setup_plot(self):
        '''Set figure size and boundaries of the point cloud,
        and setup the input callbacks'''
        # The 'figsize' parameter should be adjusted for your screen size
        self.fig = plt.figure(figsize=(10.0, 10.0))
        self.ax = self.fig.add_subplot(1, 1, 1)

        plt.tight_layout()

        # Set the limits for X (forward/backwards) and Y (side/side)
        self.ax.set_xlim(
            self.config['geometry']['W1'], self.config['geometry']['W2'])
        self.ax.set_ylim(
            self.config['geometry']['L1'], self.config['geometry']['L2'])

        self.last_point = None

        if not self.view_only:
            for k in plt.rcParams.keys():
                if k.startswith('keymap'):
                    plt.rcParams[k] = ''

            # Callbacks for button and mouse clicks
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def points_to_lines(self, P1, P2):
        '''
        Convert line from two-points format to standard form

        Params:
            P1: Nx2 numpy array of [x, y]
            P2: Nx2 numpy array of [x, y]
        Returns:
            1D numpy array of [a, b, c]
        '''
        n_lines = P1.shape[0]
        ones = np.ones((n_lines, 1))
        P1_h = np.concatenate((P1, ones), axis=1)
        P2_h = np.concatenate((P2, ones), axis=1)

        return np.cross(P1_h, P2_h)

    def point_line_distance(self, point, lines):
        '''
        Params:
            point: Nx2 numpy array of [x, y]
            lines: Nx3 numpy array of [[a, b, c], ...]
        '''
        return (np.abs(np.dot([[point[0], point[1], 1]], lines.T))
                / np.linalg.norm(lines[:, 0:2], axis=1))

    def point_line_projection(self, point, line):
        '''Transform a click from any coordinate to the
        coordinate closest to the given line'''
        l_abc = line.reshape(1, 3)
        l_ab0 = (l_abc * np.array([1, 1, 0])).reshape(1, 3)
        p = np.array([point[0], point[1], 1])

        A = np.matmul(l_ab0.T, l_abc)
        B = np.matmul(l_ab0, l_abc.T)

        P = np.eye(3) - (A / B)
        assert P.shape == (3, 3)

        proj = np.dot(P, p)
        proj /= proj[2]

        return proj[0:2]

    def add_line(self, p1, p2):
        # Convert to row frame
        R_sensor_to_row = rotation_2d(self.row_to_sensor_angle).T
        t_sensor_to_row = -R_sensor_to_row.dot(self.row_to_sensor_t)

        p1_row = R_sensor_to_row.dot(p1) + t_sensor_to_row
        p2_row = R_sensor_to_row.dot(p2) + t_sensor_to_row

        line = np.concatenate((p1_row, p2_row), axis=0)
        self.lines.append(line)

    def on_click(self, event):
        point = (event.xdata, event.ydata)

        rows = self.transform_rows(
                self.rows, self.row_to_sensor_t,
                self.row_to_sensor_angle)

        # Convert the guide lines to standard form
        row_lines = self.points_to_lines(
                rows[:, 0:2], rows[:, 2:4])

        # Determine distance from the mouse-clicked point and each line
        dists = self.point_line_distance(point, row_lines)

        # Choose the guide line that the mouse click was closest to
        i = np.argmin(dists)

        click = self.point_line_projection(
                (event.xdata, event.ydata),
                row_lines[i])

        # On a left click, store the point
        if event.button == 1:
            if self.last_point is None:
                self.last_point = click
            else:
                self.add_line(click, self.last_point)
                self.last_point = None

        # On a right click, delete the line
        if event.button == 3:
            del self.lines[-1]

        self.redraw()

    def on_key(self, event):
        # On a key press, move or rotate the lines or guides accordingly
        INC_AMOUNT_T = 0.01

        # Move left and right
        if event.key == 'a':
            self.row_to_sensor_t -= np.array([INC_AMOUNT_T*2, 0.])
        if event.key == 'd':
            self.row_to_sensor_t += np.array([INC_AMOUNT_T*2, 0.])

        # Move up and down
        if event.key == 'w':
            self.row_to_sensor_t += np.array([0., INC_AMOUNT_T])
        if event.key == 's':
            self.row_to_sensor_t -= np.array([0., INC_AMOUNT_T])

        # Rotate the yaw counter-clockwise and clockwise
        if event.key == 'r':
            self.row_to_sensor_angle += radians(.15)
        if event.key == 'f':
            self.row_to_sensor_angle -= radians(.15)

        # Increase and decrease the row spacing
        if event.key == 't':
            self.row_spacing += INC_AMOUNT_T
        if event.key == 'g':
            self.row_spacing -= INC_AMOUNT_T

        # Make the rows longer and shorter
        if event.key == 'y':
            self.row_radius += INC_AMOUNT_T * 5
        if event.key == 'h':
            self.row_radius -= INC_AMOUNT_T * 5

        self.last_point = None
        self.redraw()

    def transform_rows(self, lines, t, angle):
        R = rotation_2d(angle)
        coords = lines.reshape(-1, 2).T
        coords = np.matmul(R, coords).T + t
        lines = coords.reshape(-1, 4)

        return lines

    def draw_row_guides(self):
        self.ax.lines = []

        lines = self.transform_rows(
                self.rows, self.row_to_sensor_t,
                self.row_to_sensor_angle)

        for line in lines:
            pts = line.reshape(2, 2)
            self.draw_line(pts[:, 0], pts[:, 1], color='grey')

    def draw_line(self, x, y, color=None):
        lines = self.ax.plot(x, y, c=color)

        for line in lines:
            self.ax.draw_artist(line)

    def redraw(self):
        '''
        Efficiently update the line annotations by restoring the
        pre-rendered background lidar points all at once and then
        drawing lines on top
        '''
        # Re-calculate the rows in case they are changed
        # by the keyboard callback
        self.rows = np.array([
            [-self.row_radius, n*self.row_spacing/2.,
             self.row_radius, n*self.row_spacing/2.]
            for n in range(-7, 7+2, 2)
        ])

        self.fig.canvas.restore_region(self.background)
        self.draw_row_guides()

        # Re-draw the lines that are already labeled
        lines = self.get_lines()
        for line in lines:
            pts = line.reshape(2, 2)
            self.draw_line(pts[:, 0], pts[:, 1], color='r')

        self.fig.canvas.blit(self.fig.bbox)

    def get_labels(self):
        '''
        Retrieve the labels after drawing is completed

        Returns:
            arr: the labels in 'xyxy' format
        '''
        lines = self.get_lines()

        if len(lines) == 0:
            return np.array([])
        else:
            label_format = np.stack(
                (lines[:, 0], lines[:, 2], lines[:, 1], lines[:, 3]),
                axis=1
            )
        return label_format

    def get_lines(self):
        if len(self.lines) == 0:
            return []
        else:
            lines = self.transform_rows(
                np.stack(self.lines, axis=0),
                self.row_to_sensor_t,
                self.row_to_sensor_angle)
            return lines

    def plot_points(self, pts):
        '''
        Plot a point cloud - for use in the interactive labeler

        Parameters:
            pts (arr): the raw point cloud
        '''
        self.setup_plot()

        # Plot the Lidar points
        self.ax.scatter(
            pts[:, 0], pts[:, 1], c=pts[:, 2], vmin=-1,
            vmax=1, s=1., cmap='jet'
        )

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.redraw()

        plt.show()

    def plot_points_and_labels(self, pts, labels):
        '''
        Plot a point cloud and lines
        (for use in viewing the labels and points as one image (read-only))

        Parameters:
            pts (arr): the raw point cloud
            labels (arr): an array of arrays of tuples containing Xs and Ys
        '''
        self.setup_plot()

        # Plot the Lidar points
        self.ax.scatter(
            pts[:, 0], pts[:, 1], c=pts[:, 2], vmin=-1,
            vmax=1, s=1., cmap='jet'
        )

        # Plot the labels
        for label in labels:
            self.ax.plot(label[0], label[1], 'r')

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        plt.show()
