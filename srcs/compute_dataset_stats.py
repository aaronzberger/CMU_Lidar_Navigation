from dataset import find_reg_target_std_and_mean
import numpy as np
import os

from config import data_dir

if __name__ == '__main__':
    means, stds = find_reg_target_std_and_mean()

    # print('means', means)
    # print('stds', stds)

    stats_path = os.path.join(data_dir, 'statistics.npz')
    np.savez(stats_path, target_mean=means, target_std=stds)
