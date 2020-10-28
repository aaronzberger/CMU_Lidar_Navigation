import numpy as np
from matplotlib import pyplot as plt

# Adapted from
# https://stackoverflow.com/questions/33569626/matplotlib-responding-to-click-events
# https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
class Labeler(object):
    def __init__(self):
        self.setup_plot()

        self.lines = []
        self.last_point = None


    def setup_plot(self):
        # Sized so that the plot will not be larger than the avaliable
        # viewport on John's laptop. Issues may result on other screen sizes
        self.fig = plt.figure(figsize=(9.8, 9.8))
        self.ax = self.fig.add_subplot(1, 1, 1)

        plt.tight_layout()
        # Optimized for the lidar scans taken from the warthog robot
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(-10, 10)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


    def on_key(self, event):
        if event.key == u'left':
            print 'left'

        if event.key == u'right':
            print 'right'

    def draw_line(self, x, y):
        line = self.ax.plot(x,y)
        self.redraw()

    def redraw(self):
        # Efficiently update the line annotations by restoring the pre-rendered
        # background lidar points all at once and then drawing lines on top
        self.fig.canvas.restore_region(self.background)
        for l in self.ax.lines:
            self.ax.draw_artist(l)
        self.fig.canvas.blit(self.fig.bbox)

    def update_lines(self):
        self.lines = np.array(
            [np.concatenate((l.get_xdata(), l.get_ydata()))
            for l in self.ax.lines])

        print self.lines

    def on_click(self, event):
        if event.button == 1:
            if self.last_point is None:
                self.last_point = (event.xdata, event.ydata)
            else:
                x = [event.xdata, self.last_point[0]]
                y = [event.ydata, self.last_point[1]]
                self.last_point = None

                self.draw_line(x, y)

        if event.button == 3:
            self.ax.lines.remove(self.ax.lines[-1])
            self.redraw()

        self.update_lines()

    def plot_points(self, pts):
        self.setup_plot()

        self.ax.scatter(
            pts[:,0], pts[:,1], c=pts[:,2], vmin=-1,
            vmax=1, s=1., cmap='jet'
        ) 

        # Draw the lidar points and save them for efficient line annotation
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        plt.show()
