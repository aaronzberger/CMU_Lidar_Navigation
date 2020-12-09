import sys
sys.path.append("/home/aaron/ag_lidar_navigation-bev/srcs/models/")

import logging
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import random
import math
from torch.multiprocessing import Pool
import cv2 as cv
import os

from loss import ClassificationLoss, EmbeddingLoss
from dataset import get_data_loader
from unet import UNet
from utils import get_model_name, load_config, get_writer, plot_pr_curve
from postprocess import compute_line_matches, compute_ap, extract_lines, compute_precision_recall
from torchvision.utils import make_grid
from shapely.geometry import LineString
from tqdm import tqdm
from helpers import mkdir_p

from sklearn.decomposition import PCA
from bev import BEV


def build_model(config, device, output="class", train=True):
    '''
    Build the U-Net model
    
    Parameters:
        config (dictionary): dictionary of hyperparameter names and values for configuration
        device (torch.device): the device on which to run the network
        output (string): the type of output of the model
        train (bool): whether the model is being used for training
        
    Returns:
        net (torch.nn.Module): the network
        loss_fn (class): the loss function for the network
        optimizer (torch.optim): an optimizer (if train=True)
        scheduler (torch.optim): a scheduler (if train=True)
    '''
    if output == "class":
        out_channels = 1
    else:
        out_channels = config["embedding_dim"]
        
    net = UNet(config['geometry'], use_batchnorm=config['use_bn'], output_dim=out_channels,
            feature_scale=config["layer_width_scale"])

    # Determine the loss function to be used (if you're not sure, use ClassificationLoss)
    if output == "class":
        loss_fn = ClassificationLoss(device, config, num_classes=1)
    elif output == "embedding":
        loss_fn = EmbeddingLoss(device, config, embedding_dim=config['embedding_dim'])

    # Determine whether to run on multiple GPUs
    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("Using %s GPUs" % torch.cuda.device_count())
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)
    
    if not train:
        return net, loss_fn

    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    scheduler = None

    return net, loss_fn, optimizer, scheduler


def save_images(exp_name, input, label_map, prediction, pcl_filename, truth_filename, pred_filename):
    '''
    Save images of the input Point Cloud, the ground truth, and the U-Net prediction
    
    Parameters:
        input (Tensor): input to the network
        label_map (Tensor): labels, retrieved from a data loader
        prediction (Tensor): output from the network
        pcl_filename (string): file in which to save the point cloud
        truth_filename (string): file in which to save the ground truth
        truth_filename (string): file in which to save the prediction
    '''    
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)

    # If not already created, make a directory to save the images
    mkdir_p(config['image_save_dir'])
        
    # Save an image of the ground truth lines
    # With the first ground truth data in the batch, convert channel order: 
    # CxWxH to WxHxC and convert to grayscale image format (0-255 and 8-bit int)
    truth = np.array(label_map.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
    image_truth = cv.cvtColor(truth, cv.COLOR_GRAY2BGR)
    cv.imwrite(os.path.join(config['image_save_dir'], truth_filename), image_truth)

    # Save an image of the point cloud:
    pcl = np.array(input.cpu() * 255, dtype=np.uint8).transpose(2, 0, 1)
    image_pcl = np.amax(pcl, axis=0)
    cv.imwrite(os.path.join(config['image_save_dir'], pcl_filename), image_pcl)

    # Save an image of the U-Net prediction
    prediction_gray = np.array(prediction.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
    prediction_bgr = cv.cvtColor(prediction_gray, cv.COLOR_GRAY2BGR)
    cv.imwrite(os.path.join(config['image_save_dir'], pred_filename), prediction_bgr)
    

def validation_round(net, loss_fn, device, exp_name, epoch_num):
    net.eval()
    
    # Load hyperparameters
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)

    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
        batch_size, config['geometry'])
    
    with torch.no_grad():
        # This will keep track of total average loss for all test data
        val_loss = 0
        
        image_num = 0
        
        with tqdm(total=num_val, desc='Validation: ', unit=' pointclouds') as progress:
            for input, label_map, _, _, _ in test_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                # Forward Prop
                predictions = net(input)

                loss = loss_fn(predictions, label_map)
                
                # Update the progress bar with the current batch loss
                progress.set_postfix(**{'batch loss':"{:.4f}".format(abs(loss.item()))})
                val_loss += abs(loss.item())
                
                # To better visualize the images, exagerate the difference between 0 and 1
                predictions = torch.sigmoid(predictions)

                # After some epochs, save an image of the input, output, and ground truth for visualization
                if config['visualize']:
                    if epoch_num + 1 in config['vis_after_epoch'] or config['vis_every_epoch']:
                            truth_filename = "epoch_%s__image_%s_truth.jpg" % (epoch_num, image_num)
                            pcl_filename = "epoch_%s__image_%s_point_cloud.jpg" % (epoch_num, image_num)
                            prediction_filename = "epoch_%s__image_%s_unet_output.jpg" % (epoch_num, image_num)

                            save_images(exp_name, input[0], label_map[0], predictions[0], pcl_filename, truth_filename, prediction_filename)
                image_num += batch_size
                            
                # Update the progress bar, moving it along by one batch size
                progress.update(input.shape[0])
                
        val_loss = val_loss / len(test_data_loader)
        print("Validation Round Loss: %s" % val_loss)


def train(exp_name, device, output):
    '''
    Train the network.
    
    Parameters:
        exp_name (string): the name of the experiment for which to load the config file
        device (torch.device): the device on which to run
        output (string): the type of output for the network {class, embedding}
    '''
    # Load Hyperparameters
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)
    
    print('''\nLoaded hyperparameters:
    Learning Rate:   %s
    Batch size:      %s
    Epochs:          %s
    Device:          %s
    ''' % (learning_rate, batch_size, max_epochs, device.type if device.type == 'cpu' else "GPU X %s" % torch.cuda.device_count()))
    
    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            batch_size, config['geometry'])

    # Build the model
    net, loss_fn, optimizer, scheduler = build_model(config, device, output, train=True)
    
    print('''\nBuilt model:
    Loss Function:   %s
    Optimizer:       %s
    Scheduler:       %s
    ''' % ("Classification Loss" if isinstance(loss_fn, ClassificationLoss) else "Embedding Loss", "Adam", "None"))

    # Tensorboard Logger
    train_writer = get_writer(config, 'train')
    val_writer = get_writer(config, 'val')

    # For picking up training at the epoch where you left off. Edit this setting in config file.
    if config['resume_training']:
        start_epoch = config['resume_from']
        saved_ckpt_path = get_model_name(config)

        if config['mGPUs']:
            net.module.load_state_dict(torch.load(saved_ckpt_path, map_location=device))
        else:
            net.load_state_dict(torch.load(saved_ckpt_path, map_location=device))

        print("Successfully loaded trained checkpoint at {}".format(saved_ckpt_path))
    else:
        start_epoch = 0

    step = 1 + start_epoch * len(train_data_loader)

    # Do an initial validation round as a benchmark
    validation_round(net, loss_fn, device, exp_name, start_epoch)

    # Train for max_epochs epochs
    for epoch in range(start_epoch, max_epochs):
        
        epoch_loss = 0
        net.train()

        with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch+1, max_epochs), unit=' pointclouds') as progress:
            for input, label_map, instance_map, num_instances, image_id in train_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                optimizer.zero_grad()

                # Create mask
                mask = (0.1 * (1 - label_map)) + (label_map * 0.9)

                # Forward Prop
                predictions = net(input)
                
                # Calculate loss for this batch
                if output == "class":
                    loss = loss_fn(predictions, label_map, mask)
                elif output == "embedding":
                    loss = loss_fn(predictions, instance_map, num_instances)

                # Update the progress bar this this batch's loss
                progress.set_postfix(**{'loss (batch)': loss.item()})

                # Back Prop
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                    
                # Update the progress bar by moving it along by this batch size
                progress.update(input.shape[0])

        # Record Training Loss
        epoch_loss = epoch_loss / len(train_data_loader)
        train_writer.add_scalar('loss_epoch', epoch_loss, epoch + 1)
        print("Epoch {}: Training Loss: {:.5f}".format(
            epoch + 1, epoch_loss))
        
        # Run Validation
#         if (epoch + 1) % 2 == 0:
#             tic = time.time()
#             val_metrics, _, _, log_images = eval_batch(config, net, loss_fn, test_data_loader, device)

#             for tag, value in val_metrics.items():
#                 val_writer.add_scalar(tag, value, epoch + 1)

#             # val_writer.image_summary('Predictions', log_images, epoch + 1)
#             print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(
#                 epoch + 1, time.time() - tic, val_metrics['loss']))

        # Save Checkpoint
        if (epoch + 1) == max_epochs or (epoch + 1) % config['save_every'] == 0:
            model_path = get_model_name(config, epoch + 1)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print("Checkpoint for epoch {} saved at {}\n".format(epoch + 1, model_path))

        if scheduler is not None:
            scheduler.step()

    print('Finished Training')
    
    
def eval_loader(config, net, loss_fn, loader, loader_name, device):
    '''
    Evaluate the performance of the model on the specified data loader
    
    Parameters:
        config (dictionary): dictionary of hyperparameter names and values for configuration
        net (torch.nn.Module): the network
        loss_fn (class): the loss function for the network
        loader (Dataset): a data loader in which to retrieve the input
        loader_name (string): the name to give the data loader (only for printing)
        device (torch.device): the device on which to run the network

    Returns:
        dict: dictionary of name-value pairs (useful information on the evaluation)
    '''
    net.eval()
    
    forward_pass_times = []
    losses = []

    with torch.no_grad():
        with tqdm(total=len(loader), desc='Evaluate %s Data: ' % loader_name, unit=' pcl') as progress:
            for i, data in enumerate(loader):
                input, label_map, _, _, image_id = data

                input = input.to(device)
                label_map = label_map.to(device)

                start_time = time.time()
                
                # Forward Prop
                predictions = net(input)

                forward_pass_times.append(time.time() - start_time)

                loss = loss_fn(predictions, label_map)
                
                losses.append(loss.item())
                
                # Update the progress bar with the current batch loss
                progress.set_postfix(**{'loss':"{:.4f}".format(abs(loss.item()))})

                # Update the progress bar, moving it along by one batch size
                progress.update(input.shape[0])

    metrics = {}

    metrics['time_mean'] = np.mean(forward_pass_times)
    metrics['time_max'] = np.amax(forward_pass_times)
    metrics['time_min'] = np.amin(forward_pass_times)
    metrics['time_std'] = np.std(forward_pass_times)
    
    metrics['loss_mean'] = np.mean(losses)
    metrics['loss_max'] = np.amax(losses)
    metrics['loss_min'] = np.amin(losses)
    metrics['loss_std'] = np.std(losses)
    
    return metrics


def evaluate_model(exp_name, device, plot=True):
    '''
    Determine the total performance of the network on all data
    
    Parameters:
        exp_name (string): the name of the experiment for which to load the config file
        device (torch.device): the device on which to run
        plot (bool): whether to plot the results for visualization
    '''
    # Load Hyperparameters
    config, _, _, _ = load_config(exp_name)
    
    # Build the model
    net, loss_fn = build_model(config, device, train=False)

    saved_ckpt_path = get_model_name(config)

    if config['mGPUs']:
        net.module.load_state_dict(torch.load(saved_ckpt_path, map_location=device))
    else:
        net.load_state_dict(torch.load(saved_ckpt_path, map_location=device))

    print("Successfully loaded trained checkpoint at {}".format(saved_ckpt_path))
    
    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            1, config['geometry'])

    # Evaluate the performance on the training data
    metrics_train = eval_loader(config, net, loss_fn, train_data_loader, "Train", device)

    # Evaluate the performance on the testing data
    metrics_val = eval_loader(config, net, loss_fn, test_data_loader, "Test", device)
    
    total_inputs = len(train_data_loader) + len(test_data_loader)
    time_mean = ((metrics_train['time_mean'] * len(train_data_loader)) + (metrics_val['time_mean'] * len(test_data_loader))) / total_inputs
    
    print('''\n
    Network:
        Weights:         {}
        Loss Function:   {}
    
    Time Evaluation:
        Mean:            {:.4f} seconds per point cloud
        Max:             {:.4f} seconds per point cloud
        Min:             {:.4f} seconds per point cloud
    
    Training Data Loss:
        Mean:            {:.4f}
        Max:             {:.4f}
        Min:             {:.4f}
    
    Validation Data Loss:
        Mean:            {:.4f}
        Max:             {:.4f}
        Min:             {:.4f}
    
    '''.format(saved_ckpt_path, "Classification Loss" if isinstance(loss_fn, ClassificationLoss) else "Embedding Loss", time_mean, max(metrics_train['time_max'], metrics_val['time_max']), min(metrics_train['time_min'], metrics_val['time_min']), metrics_train['loss_mean'], metrics_train['loss_max'], metrics_train['loss_min'], metrics_val['loss_mean'], metrics_val['loss_max'], metrics_val['loss_min']))
    
    #fig_name = "PRCurve_val_" + config['name']
    #legend = "AP={:.1%}".format(val_metrics['AP'])
    #plot_pr_curve(val_precisions, val_recalls, legend, name=fig_name)

def eval_one(net, loss_fn, loader, image_id, device):
    '''
    Pass one point cloud through the network
    
    Parameters:
        net (torch.nn.Module): the network, with a loaded dict
        loss_fn (class): a loss function to evaluate the prediction
        loader (Dataset): a data loader in which to retrieve the input
        image_id (int): the image_id in the loader provided
        device (torch.device): the device on which to run
    
    Returns:
        torch.Tensor: the input (point cloud)
        torch.Tensor: the ground truth lines
        torch.Tensor: the U-Net prediction
        float: the loss calculated
        float: the time taken to complete the forward pass
    '''
    # Retrieve this specific item in the data loader
    input, label_map, _, _, image_id = loader.dataset[image_id]
    
    input = input.to(device).unsqueeze(0)
    label_map = label_map.to(device).unsqueeze(0)

    # Forward Pass
    start_time = time.time()
    prediction = net(input)
    forward_pass_time = time.time() - start_time

    loss = loss_fn(prediction, label_map)

    return input[0], label_map, prediction, loss.item(), forward_pass_time


def test(exp_name, device, image_id):
    '''
    Test the network by using one image from the testing dataset, and save images of the process.
    
    Parameters:
        exp_name (string): the name of the experiment for which to load the config file
        device (torch.device): the device on which to run
        image_id (int): the image_id in the test data loader
    '''
    # Load Hyperparameters
    config, _, _, _ = load_config(exp_name)
    
    # Build the model
    net, loss_fn = build_model(config, device, train=False)
    
    # Load the weights of the network
    net.load_state_dict(torch.load(get_model_name(config), map_location=device))
    
    print('''\nBuilt model:
    Loss Function:   %s
    Weights:         %s
    ''' % ("Classification Loss" if isinstance(loss_fn, ClassificationLoss) else "Embedding Loss", get_model_name(config)))
    
    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            config['batch_size'], config['geometry'])    
    
    net.eval()

    with torch.no_grad():
        input, label_map, prediction, loss, time = eval_one(net, loss_fn, test_data_loader, image_id, device)
        
    truth_filename = "EVAL_num_%s_truth.jpg" % image_id
    pcl_filename = "EVAL_num_%s_point_cloud.jpg" % image_id
    prediction_filename = "EVAL_num_%s_unet_output.jpg" % image_id

    save_images(exp_name, input, label_map[0], prediction[0], pcl_filename, truth_filename, prediction_filename)        

    print('''\nEvaluated Image %s:
    Forward Pass Time:   %s
    Loss:                %s
    Ground Truth Image   %s
    Point Cloud Image    %s
    U-Net Output Image   %s
    ''' % (image_id, time, loss, os.path.join(config['image_save_dir'], truth_filename), os.path.join(config['image_save_dir'], pcl_filename), os.path.join(config['image_save_dir'], prediction_filename)))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-Net training module')
    parser.add_argument('mode', choices=['train', 'eval', 'test'], help='mode for the model')
    parser.add_argument('--name', required=True, help="name of the experiment")
    parser.add_argument('--test_id', type=int, default=25, help="id of the image to test")
    parser.add_argument('--output', required=True, help="output of the model")
    args = parser.parse_args()

    if args.output not in ["class", "embedding"]:
        raise ValueError("output must be one of {class, embedding}")

    # Choose a device for the model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   

    if args.mode=='train':
        train(args.name, device, args.output)
    if args.mode=='eval':
        evaluate_model(args.name, device, plot=False)
    if args.mode=='test':
        test(args.name, device, image_id=args.test_id)