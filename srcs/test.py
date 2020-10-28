import matplotlib.pyplot as plt
from matplotlib import cm

import random
import os

import torch
from sys import argv

from utils import load_config
from models.unet import UNet
from bev import BEV

from torch.utils.data import DataLoader
from dataset import VineyardDataset
from sklearn.cluster import MeanShift, DBSCAN, KMeans

from fit_line import fit_line
from postprocess import extract_lines, compute_line_matches, compute_precision_recall

import cv2
import numpy as np

import pathlib

if len(argv) != 3:
    print("Usage: test.py SPLIT CHECKPOINT")

split = argv[1]
path = argv[2]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


config, _, _, _ = load_config("10meter_unet_quarterwidth")
geom = config["geometry"]

bev = BEV(geom)

model = UNet(config['geometry'], use_batchnorm=config['use_bn'], output_dim=1,
        feature_scale=config["layer_width_scale"])
model.to(device)

model.load_state_dict(torch.load(path, map_location=device))

checkpoint_name = "_".join(path.split("/")[1:])
print(checkpoint_name)

results_dir = os.path.join("results", checkpoint_name, split)
pathlib.Path(results_dir).mkdir(parents=True,
    exist_ok=True)

random.seed(0)
torch.manual_seed(0)

batch_size = 2
test_dataset = VineyardDataset(
        config["geometry"], split=split)
test_data_loader = DataLoader(test_dataset, 
        batch_size=batch_size, num_workers=8, shuffle=False)

all_pred_matches = []
num_gts = 0
num_preds = 0

for n, (inp, target, ins_target, n_ins, idx) in enumerate(test_data_loader):
    if n < (284 / 2):
        continue 

    inp, target = inp.to(device), target.to(device)
    preds = model(inp)
    preds = torch.sigmoid(preds)

    for i in range(len(inp)):

        pred_i = preds[i,...].permute(1, 2, 0)
        inp_i = inp[i,...].permute(1, 2, 0)
        target_i = target[i,...].permute(1, 2, 0)

        inp_i = inp_i.detach().cpu().numpy()
        pred_i = pred_i.detach().cpu().numpy()[:,:,0]
        target_i = target_i.detach().cpu().numpy()[:,:,0]

        img = BEV.make_bev_viz_image(inp_i)

        plt.clf()

        plt.subplot(2, 2, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img)

        plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(target_i)

        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(pred_i)
        
        lines = extract_lines(pred_i, min_ratio=10)
        gt_lines = test_dataset.get_labels(idx[i])

        # print(lines)
        # print(gt_lines)

        num_gts += len(gt_lines)
        num_preds += len(lines)

        gt_matches, pred_matches, distances = compute_line_matches(gt_lines,
                lines, distance_threshold=2.5)

        # print(pred_matches)

        all_pred_matches.extend(pred_matches)

        plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img)

        for j, (x1, y1, x2, y2) in enumerate(gt_lines):
            if gt_matches[j] != -1:
                # True Positive
                x1, y1, x2, y2 = lines[gt_matches[j]]
                plt.plot([y1, y2], [x1, x2], c='b', marker='o', markersize=1, linewidth=0.5)

                plt.plot([y1, y2], [x1, x2], c='g', marker='o', markersize=1,linewidth=0.5)

        for j, (x1, y1, x2, y2) in enumerate(gt_lines):
            if gt_matches[j] == -1:
                # False Negative
                plt.plot([y1, y2], [x1, x2], c='y', marker='o', markersize=1,linewidth=0.5)
        
        for j, (x1, y1, x2, y2) in enumerate(lines):
            # False Positive
            if pred_matches[j] == -1:
                plt.plot([y1, y2], [x1, x2], c='r', marker='o', markersize=1,linewidth=0.5)


        name = f"{(n*batch_size)+i}.png"
        print(name)

        # plt.savefig(os.path.join(results_dir, name), dpi=200)
        # plt.show()

        pts = test_dataset.get_pointcloud(idx[i], frame='bev')
        bev.visualize_lines_3d([], pts, order='xyxy')

    # break

all_pred_matches = np.array(all_pred_matches)

precision, recall, tp, fp, fn = compute_precision_recall(all_pred_matches,
        num_gts, num_preds)

results_str = f"""
Precision: {precision}
Recall: {recall}
TP: {tp}
FP: {fp}
FN: {fn}
"""

print(results_str)

with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
    f.write(results_str)
