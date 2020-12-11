'''
This file is not run independently

Helper functions
'''

import errno
import json
import os
import math
from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import base_dir, exp_name


def load_config(exp_name):
    '''
    Load the configuration file

    Parameters:
         name: the name of the 'experiment'
    Returns:
         config (dict): Python dictionary of hyperparameter name-value pairs
         learning_rate (float): the learning rate of the optimzer
         batch_size: batch size used during training
         max_epochs: number of epochs to train the network fully
    '''

    path = os.path.join(base_dir, 'experiments', exp_name, 'config.json')
    
    with open(path) as file:
        config = json.load(file)

    assert config['name']==exp_name

    return config, config['learning_rate'], config['batch_size'], config['max_epochs']

def get_model_name(config, epoch=None):
    '''
    Generate a path to the relevant state dictionary for the model to load

    Parameters:
        config (dict): the configuration file
        epoch (int): the epoch to resume from
    Returns:
        string: a path to the state dictionary
    '''

    name = config['name']

    state_dict_folder = os.path.join(base_dir, 'experiments', name, 'state_dicts')
    
    if not os.path.exists(state_dict_folder):
        mkdir_p(state_dict_folder)
        
    if epoch is None or not os.path.exists(os.path.join(state_dict_folder, str(epoch)+'epoch')):
        epoch = config['resume_from']
    
    return os.path.join(state_dict_folder, str(epoch)+'epoch')


def load_target_mean_std():
    config, _, _, _ = load_config(exp_name)
    stats_path = os.path.join(config['data_dir'], 'statistics.npz')
    data = np.load(stats_path)
    
    return data['target_mean'], data['target_std']

def trasform_label2metric(label, ratio=4, grid_size=0.1, base_height=100):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''

    metric = np.copy(label)
    metric[..., 1] -= base_height
    metric = metric * grid_size * ratio

    return metric

def transform_metric2label(metric, ratio=4, grid_size=0.1, base_height=100):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in metric space
    :return: numpy array of shape [..., 2] of the same coordinates in label_map space
    '''

    label = (metric / ratio ) / grid_size
    label[..., 1] += base_height
    return label

def get_writer(config, mode='train'):
    folder = os.path.join('logs', config['name'], mode)
    if not os.path.exists(folder):
        os.makedirs(folder)

    return SummaryWriter(folder)


def plot_pr_curve(precisions, recalls, legend, name='PRCurve'):

    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, '.')
    ax.set_title('Precision Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend([legend], loc='upper right')
    path = os.path.join('Figures', name)
    fig.savefig(path)
    print('PR Curve saved at', path)

def get_points_in_a_rotated_box(corners, label_shape=[200, 175]):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin = 0
    ymin = 0
    xmax = label_shape[1]
    ymax = label_shape[0]

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels


def writefile(config, filename, value):
    path = os.path.join('experiments', config['name'], filename)
    with open(path, 'a') as f:
        f.write(value)

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())


def mkdir_p(path):
    '''
    Create a directory at a given path if it does not already exist

    Parameters:
        path (string): the full os.path location for the directory
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rotation_2d(theta):
    return  np.array(
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]])

if __name__ == '__main__':
    maskFOV_on_BEV(0)
