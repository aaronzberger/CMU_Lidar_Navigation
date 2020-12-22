'''
This file is not run independently

Helper functions
'''

import errno
import json
import os
from math import sin, cos

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

    assert config['name'] == exp_name

    return config, config['learning_rate'], \
        config['batch_size'], config['max_epochs']


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

    state_dict_folder = os.path.join(
        base_dir, 'experiments', name, 'state_dicts')

    if not os.path.exists(state_dict_folder):
        mkdir_p(state_dict_folder)

    if epoch is None or not os.path.exists(
            os.path.join(state_dict_folder, str(epoch)+'epoch')):
        epoch = config['resume_from']

    return os.path.join(state_dict_folder, str(epoch)+'epoch')


def load_target_mean_std():
    config, _, _, _ = load_config(exp_name)
    stats_path = os.path.join(config['data_dir'], 'statistics.npz')
    data = np.load(stats_path)

    return data['target_mean'], data['target_std']


def get_writer(config, mode='train'):
    folder = os.path.join('logs', config['name'], mode)
    if not os.path.exists(folder):
        os.makedirs(folder)

    return SummaryWriter(folder)


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
    return np.array(
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]])
