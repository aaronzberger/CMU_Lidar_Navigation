import os
import csv

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from config import exp_name
from bev import BEV # pointcloud_to_bev, visualize_bev, make_targets
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
#             print('Loading target mean and standard deviation')
            self.target_mean, self.target_std = load_target_mean_std()
        else:
            self.target_mean, self.target_std = \
                    np.zeros(self.reg_map_channels),\
                    np.ones(self.reg_map_channels)
#             print('Not normalizing regression map')

    def __len__(self):
        return len(self.dataset)

    def get_pointcloud(self, item, frame='bev'):
        if frame != 'bev' and frame != 'orig':
            raise ValueError('frame must be one of {frame, orig}')

        raw_path, _ = self.dataset[item]
        raw_data = np.load(raw_path)

        pts = self.bev.standardize_pointcloud(raw_data['pointcloud'][:,0:3])
        
        if frame == 'bev':
            pts -= self.bev.mins
            pts /= self.bev.resolutions[0]

        return pts


    def get_labels(self, item, order='xyxy', frame='bev'):
        if order != 'xxyy' and order != 'xyxy':
            raise ValueError('order must be one of {xxyy, xyxy}')

        if frame != 'bev' and frame != 'orig':
            raise ValueError('frame must be one of {frame, orig}')

        _, label_path = self.dataset[item]
        label_data = np.load(label_path)
        labels = label_data['labels']
      
        labels_xxyy = self.bev.crop_labels(labels)

        if frame == 'bev':
            labels_xxyy = self.bev.labels_to_bev(labels_xxyy)

        if len(labels) > 0:
            if order == 'xxyy':
                return labels_xxyy
            else:
                return labels_xxyy[:,[0,2,1,3]]
        else:
            return np.empty((0, 4))
        

    def __getitem__(self, item):
        raw_path, label_path = self.dataset[item]
        
        label_data = np.load(label_path)
        raw_data = np.load(raw_path)

        pts = raw_data['pointcloud'][:,0:3]
        labels = label_data['labels']
        
        _, label_map, instance_map, num_instances = self.bev.make_targets(labels)

        bev = self.bev.pointcloud_to_bev(pts)

        bev = torch.from_numpy(bev).permute(2, 0, 1)
        label_map = torch.from_numpy(label_map).unsqueeze(0)

        instance_map = torch.from_numpy(instance_map)
        
        return bev, label_map, instance_map, num_instances, item


    def reg_target_normalize(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std

    def reg_target_denormalize(self, reg_map):
        # cls_map = label_map[..., 0]
        # reg_map = label_map[..., 1:]

        #index = np.nonzero(cls_map)
        reg_map = (reg_map * self.target_std) + self.target_mean

    
    def load_dataset(self, split):
        config, _, _, _ = load_config(exp_name)
        split_file = os.path.join(config['data_dir'], split + '.csv')

        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            dataset = []
            for row in reader:
                dataset.append((row['raw'], row['label']))
#CAUSED ERRORS IN EXECUTION
            # print(f'Found {len(dataset)} examples')

        return dataset

def get_data_loader(batch_size, geometry, shuffle_test=False):
    '''
    Retrieve the DataLoaders for training and testing
    
    Parameters:
        batch_size (int): the batch_size for both datasets
        geometry (arr): the geometry with which to pre-process the point clouds
        
    Returns:
        DataLoader: for training
        DataLoader: for testing/validation
        int: length of the training dataset
        int: length of the testing/validation dataset
    '''
    train_dataset = VineyardDataset(
            geometry, split='train')
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=3)

    val_dataset = VineyardDataset(
            geometry, split='val')
    val_data_loader = DataLoader(val_dataset, shuffle=shuffle_test, batch_size=batch_size, num_workers=8)
    
    return train_data_loader, val_data_loader, len(train_dataset), len(val_dataset)