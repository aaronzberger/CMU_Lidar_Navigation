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

def find_instance_num_distribution(config_name):
    config, _, _, _ = load_config(config_name)
    datasets = [
        VineyardDataset(config['geometry'], split=split) 
        for split in VineyardDataset.splits]
    
    counts = [[] for _ in datasets] 
    for count, dataset in zip(counts, datasets): 
        for _, _, _, num_instances, _ in dataset:
            count.append(num_instances)

    counts = [np.bincount(count) for count in counts]

    np.set_printoptions(precision=2, suppress=True)
    for name, counts in zip(VineyardDataset.splits, counts):
        print(name, counts, counts / np.sum(counts))

def find_reg_target_std_and_mean():
    config, _, _, _ = load_config('default')
    dataset = VineyardDataset(config['geometry'], normalize_reg_map=False)  
    reg_targets = [[] for _ in range(dataset.reg_map_channels)]

    for _, label_map, _, _, _ in dataset:
        locs = torch.where(label_map[0,:,:] == 1)
        
        for j in range(1, dataset.reg_map_channels + 1):
            m = label_map[j,:,:]
            reg_targets[j-1].extend(list(m[locs]))

    print([len(l) for l in reg_targets])
    reg_targets = np.array(reg_targets)
    print(reg_targets.shape)
    means = reg_targets.mean(axis=1)
    stds = reg_targets.std(axis=1)

    np.set_printoptions(precision=3, suppress=True)
    print('Means', means)
    print('Stds', stds)
    
    return means, stds

if __name__ == '__main__':
    config, _, _, _ = load_config('default')
    dataset = VineyardDataset(config['geometry'])

    print(dataset[0])
    print(len(dataset))

    find_instance_num_distribution('10meter')