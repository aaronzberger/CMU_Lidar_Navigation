'''
This file is not run independently

Helper functions for all classes
'''

import errno
import json
import os
from math import sin, cos

import numpy as np

from config import base_dir, exp_name
from loss.classification_loss import Classification_Loss
from loss.discriminative_loss import Discriminative_Loss
from loss.focal_loss import Focal_Loss
from loss.successive_e_c_loss import Successive_E_C_Loss
from loss.successive_e_f_loss import Successive_E_F_Loss


def load_config(exp_name):
    '''
    Load the configuration file

    Parameters:
        name: the name of the 'experiment'
    Returns:
        config (dict): Python dictionary of hyperparameter name-value pairs
        learning_rate (float): the learning rate of the optimzer
        batch_size: batch size used during training
        max_epochs: number of epochs to train the network
    '''

    path = os.path.join(base_dir, 'experiments', exp_name, 'config.json')

    with open(path) as file:
        config = json.load(file)

    assert config['name'] == exp_name

    return config, config['learning_rate'], \
        config['batch_size'], config['max_epochs']


def get_model_path(config, epoch=None):
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

    if epoch is None:
        epoch = config['resume_from']

    return os.path.join(state_dict_folder, str(epoch)+'epoch')


def load_target_mean_std():
    config, _, _, _ = load_config(exp_name)
    stats_path = os.path.join(config['data_dir'], 'statistics.npz')
    data = np.load(stats_path)

    return data['target_mean'], data['target_std']


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


def get_loss_string(loss_fn):
    if isinstance(loss_fn, Classification_Loss):
        return 'Alpha-balanced Classification Loss'
    elif isinstance(loss_fn, Focal_Loss):
        return 'Focal Loss'
    elif isinstance(loss_fn, Discriminative_Loss):
        return 'Discriminative Loss'
    elif isinstance(loss_fn, Successive_E_F_Loss):
        return 'Successive Embedding and Focal Loss'
    elif isinstance(loss_fn, Successive_E_C_Loss):
        return 'Successive Embedding and ' + \
            'Alpha-balanced Classification Loss'
    else:
        return 'Unknown'
