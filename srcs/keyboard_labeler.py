import numpy as np
from matplotlib import pyplot as plt
from math import radians, degrees

from helpers import rotation_2d

ROW_SPACING = 2.55

# Adapted from
# https://stackoverflow.com/questions/33569626/matplotlib-responding-to-click-events
# https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
class Labeler(object):
    def __init__(self):
        # self.setup_plot()
 
        # format is x0, y0, theta
        self.row_to_sensor_t = np.array([0., 0.])
        self.row_to_sensor_angle = 0

        ROW_LEN = 40
        self.rows = np.array([
            [0., n*ROW_SPACING/2., ROW_LEN, n*ROW_SPACING/2.] 
            for n in range(-7, 7+2, 2)
        ])
        self.lines = []

    def setup_plot(self):
        # Sized so that the plot will not be larger than the avaliable
        # viewport on John's laptop. Issues may result on other screen sizes
        self.fig = plt.figure(figsize=(9.8, 9.8))
        self.ax = self.fig.add_subplot(1, 1, 1)

        plt.tight_layout()
        # Optimized for the lidar scans taken from the warthog robot
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(-10, 10)

        self.last_point = None

        for k in plt.rcParams.keys():
            if k.startswith('keymap'):
                plt.rcParams[k] = ''

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def points_to_lines(self, P1, P2):
        '''
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
            point: 1D numpy array of [x, y]
            lines: Nx3 numpy array of [[a, b, c], ...]
        '''

        return (np.abs(np.dot([[point[0], point[1], 1]], lines.T))
                / np.linalg.norm(lines[:,0:2], axis=1))

    def point_line_projection(self, point, line):
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


    def add_annotation(self, p1, p2):
        """
        Add annotation (sensor frame)
        """

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

        row_lines = self.points_to_lines(
                rows[:,0:2], rows[:,2:4])
        dists = self.point_line_distance(point, row_lines)
        i = np.argmin(dists)
        row_line = rows[i]

        click = self.point_line_projection(
                (event.xdata, event.ydata),
                row_lines[i])

        if event.button == 1:
            if self.last_point is None:
                self.last_point = click
            else:
                self.add_annotation(click, self.last_point)
                self.last_point = None
                        
        if event.button == 3:
            del self.lines[-1]

        self.redraw()


    def on_key(self, event):
        INC_AMOUNT_T = 0.01
        if event.key == 'a':
            self.row_to_sensor_t -= np.array([INC_AMOUNT_T, 0.])
        if event.key == 'd':
            self.row_to_sensor_t += np.array([INC_AMOUNT_T, 0.])
        if event.key == 's':
            self.row_to_sensor_t -= np.array([0., INC_AMOUNT_T])
        if event.key == 'w':
            self.row_to_sensor_t += np.array([0., INC_AMOUNT_T])
        if event.key == 'q':
            self.row_to_sensor_angle += radians(.25)
        if event.key == 'e':
            self.row_to_sensor_angle -= radians(.25)

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
            self.draw_line(pts[:,0], pts[:,1], color='grey')


    def draw_line(self, x, y, color=None, redraw=True):
        lines = self.ax.plot(x,y, c=color)

        for line in lines:
            self.ax.draw_artist(line) 


    def redraw(self):
        # Efficiently update the line annotations by restoring the 
        # pre-rendered background lidar points all at once and then 
        # drawing lines on top

        self.fig.canvas.restore_region(self.background)
        self.draw_row_guides()

        lines = self.get_lines()
        for line in lines:
            pts = line.reshape(2, 2)
            self.draw_line(pts[:,0], pts[:,1], color='r')

        self.fig.canvas.blit(self.fig.bbox)


    # def update_lines(self):
    #     self.lines = np.array(
    #         [np.concatenate((l.get_xdata(), l.get_ydata()))
    #         for l in self.ax.lines])

    def get_labels(self):
        lines = self.get_lines()

        if len(lines) == 0:
            return np.array([])
        else:
            label_format = np.stack(
                (lines[:,0], lines[:,2], lines[:,1], lines[:,3]),
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
        self.setup_plot()

        self.ax.scatter(
            pts[:,0], pts[:,1], c=pts[:,2], vmin=-1,
            vmax=1, s=1., cmap='jet'
        ) 

        # Draw the lidar points and save them for efficient line annotation
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.redraw()

        plt.show()
