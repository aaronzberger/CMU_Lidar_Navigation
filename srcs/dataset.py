'''
This file is not run independently

Dataset generator for retreiving data to pass through the network
'''

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from bev import BEV
from config import exp_name
from utils import load_target_mean_std, load_config


class VineyardDataset(Dataset):
    splits = ['test', 'train', 'val']

    def __init__(self, geom, split='train',
                 normalize_reg_map=False):
        self.bev = BEV(geom)
        self.reg_map_channels = self.bev.reg_map_channels

        if split in self.splits:
            self.dataset = self.load_dataset(split)
        else:
            raise ValueError('Split must be one of {train, test, val}')

        if normalize_reg_map:
            self.target_mean, self.target_std = load_target_mean_std()
        else:
            self.target_mean, self.target_std = \
                    np.zeros(self.reg_map_channels),\
                    np.ones(self.reg_map_channels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        '''Return data in the correct format for the network'''
        raw_path, label_path = self.dataset[item]

        label_data = np.load(label_path)
        raw_data = np.load(raw_path)

        pts = raw_data['pointcloud'][:, 0:3]
        labels = label_data['labels']

        _, label_map, instance_map, num_instances = \
            self.bev.make_targets(labels)

        bev = self.bev.pointcloud_to_bev(pts)
        bev = torch.from_numpy(bev).permute(2, 0, 1)

        label_map = torch.from_numpy(label_map).unsqueeze(0)

        instance_map = torch.from_numpy(instance_map)

        return bev, label_map, instance_map, num_instances, item

    def load_dataset(self, split):
        '''Retrieve an array of all raw and labeled data for this split'''
        config, _, _, _ = load_config(exp_name)

        # Load the csv file containing paths to the raw and labeled data
        # Run split_data.py to generate these files
        split_file = os.path.join(config['data_dir'], split + '.csv')

        if not os.path.exists(split_file):
            print('%s data does not exists at path %s.' % (split, split_file)
                  + 'Run split_data.py to generate these files.')
            return []

        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            dataset = []
            for row in reader:
                dataset.append((row['raw'], row['label']))

        return dataset


def get_data_loader(batch_size, geometry, shuffle_test=False):
    '''
    Retrieve the DataLoaders for training and testing

    Parameters:
        batch_size (int): the batch_size for both datasets
        geometry (arr): the geometry with which to pre-process the point clouds
        shuffle_test (bool): whether to shuffle the testing data
            (if you have been saving images of testing data, set to False)

    Returns:
        DataLoader: for training
        DataLoader: for testing
        int: length of the training dataset
        int: length of the testing dataset
    '''
    train_dataset = VineyardDataset(
            geometry, split='train')
    train_data_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=batch_size, num_workers=3)

    test_dataset = VineyardDataset(
            geometry, split='test')
    test_data_loader = DataLoader(
        test_dataset, shuffle=shuffle_test,
        batch_size=batch_size, num_workers=8)

    return train_data_loader, test_data_loader, \
        len(train_dataset), len(test_dataset)
