# import sys

import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import csv

import cv2 as cv
from tqdm import tqdm

from config import base_dir, exp_name
from dataset import get_data_loader
from loss.classification_loss import Classification_Loss
from loss.focal_loss import Focal_Loss
from loss.successive_e_c_loss import Successive_E_C_Loss
from loss.successive_e_f_loss import Successive_E_F_Loss
from utils import get_model_name, load_config, get_writer
from utils import mkdir_p

from models.unet import UNet


def build_model(config, device, train=True):
    '''
    Build the U-Net model

    Parameters:
        config (dict): dict of hyperparam names and values for configuration
        device (torch.device): the device on which to run
        loss (str): determines loss function to be used
        train (bool): whether the model is being used for training

    Returns:
        net (torch.nn.Module): the network
        loss_fn (class): the loss function for the network
        optimizer (torch.optim): an optimizer (if train=True)
        scheduler (torch.optim): a scheduler (if train=True)
    '''
    net = UNet(config['geometry'], use_batchnorm=config['use_bn'],
               output_dim=1, feature_scale=config['layer_width_scale'])

    # Determine the loss function to be used
    if config['training_loss'] == 'c' or not train:
        loss_fn = Classification_Loss(config['classification_alpha'], config['reduction'])
    elif config['training_loss'] == 'f':
        loss_fn = Focal_Loss(
            device, alpha=config['focal_alpha'], gamma=config['focal_gamma'],
            reduction=config['reduction'])
    elif config['training_loss'] == 's_ef':
        loss_fn = Successive_E_F_Loss(
            device, lam=config['successive_lambda'],
            alpha=config['focal_alpha'], gamma=config['focal_gamma'],
            margin_s=config['embedding_margin_s'],
            margin_d=config['embedding_margin_d'],
            reduction=config['reduction'])
    elif config['training_loss'] == 's_ec':
        loss_fn = Successive_E_C_Loss(
            device, lam=config['successive_lambda'],
            alpha=config['classification_alpha'],
            margin_s=config['embedding_margin_s'],
            margin_d=config['embedding_margin_d'],
            reduction=config['reduction'])
    else:
        raise ValueError('loss argument must be in [c, f, s_ef, s_ec]')

    # Determine whether to run on multiple GPUs
    if config['mGPUs'] and torch.cuda.device_count > 1 and train:
        print('Using %s GPUs' % torch.cuda.device_count())
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)

    if not train:
        return net, loss_fn

    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    scheduler = None

    return net, loss_fn, optimizer, scheduler


def save_images(exp_name, input, label_map, prediction,
                pcl_filename, truth_filename, pred_filename):
    '''
    Save images of the input Point Cloud,
    the ground truth, and the U-Net prediction

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
    save_dir = os.path.join(base_dir, 'experiments', exp_name, 'images')
    mkdir_p(save_dir)

    # Save an image of the ground truth lines
    # With the first ground truth data in the batch, convert channel order:
    # CxWxH to WxHxC & convert to grayscale image format (0-255 and 8-bit int)
    truth = np.array(label_map.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
    image_truth = cv.cvtColor(truth, cv.COLOR_GRAY2BGR)
    cv.imwrite(os.path.join(save_dir, truth_filename), image_truth)

    # Save an image of the point cloud:
    pcl = np.array(input.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
    image_pcl = np.amax(pcl, axis=2)
    cv.imwrite(os.path.join(save_dir, pcl_filename), image_pcl)

    # Save an image of the U-Net prediction
    # To better visualize the images, exagerate the difference between 0 and 1
    prediction = torch.sigmoid(prediction)
    prediction_gray = np.array(
        prediction.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
    prediction_bgr = cv.cvtColor(prediction_gray, cv.COLOR_GRAY2BGR)
    cv.imwrite(os.path.join(save_dir, pred_filename), prediction_bgr)


def validation_round(net, device, exp_name, epoch_num):
    '''
    Find testing data loss and save images of the pipeline if applicable

    Parameters:
        net (torch.nn.Module): the network
        loss_fn (class): the loss function for the network
        device (torch.device): the device on which to run
        exp_name (str): the name of the config file to load
        epoch_num (int): the epoch number (for saving images)
    '''
    net.eval()

    # Load hyperparameters
    config, _, batch_size, _ = load_config(exp_name)
    
    loss_fn = Classification_Loss(config['classification_alpha'], config['reduction'])

    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
        batch_size, config['geometry'])

    with torch.no_grad():
        # This will keep track of total average loss for all test data
        val_loss = 0

        image_num = 0

        with tqdm(total=num_val, desc='Validation: ', unit=' pointclouds') \
                as progress:
            for input, label_map, _, _, _ in test_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                # Forward Prop
                predictions = net(input)

                loss = loss_fn(predictions, label_map)

                # Update the progress bar with the current batch loss
                progress.set_postfix(
                    **{'batch loss': '{:.4f}'.format(abs(loss.item()))})
                val_loss += abs(loss.item())

                # After some epochs, save an image of the
                # input, output, and ground truth for visualization
                if config['visualize']:
                    if epoch_num + 1 in config['vis_after_epoch'] or \
                            config['vis_every_epoch']:
                        truth_filename = 'epoch_%s__image_%s_truth.jpg' % \
                            (epoch_num, image_num)
                        pcl_filename = 'epoch_%s__image_%s_point_cloud.jpg' % \
                            (epoch_num, image_num)
                        prediction_filename = 'epoch_%s__image_%s_unet.jpg' % \
                            (epoch_num, image_num)

                        save_images(exp_name, input[0], label_map[0],
                                    predictions[0], pcl_filename,
                                    truth_filename, prediction_filename)
                image_num += batch_size

                # Update the progress bar, moving it along by one batch size
                progress.update(input.shape[0])

        val_loss = val_loss / len(test_data_loader)
        print('Validation Round Loss: %s' % val_loss)
        return val_loss


def train(exp_name, device):
    '''
    Train the network.

    Parameters:
        exp_name (str): the name of the config file to load
        device (torch.device): the device on which to run
    '''
    # Load Hyperparameters
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)

    if device.type == 'cpu':
        device_str = 'CPU'
    elif not config['mGPUs']:
        device_str = 'GPU X 1'
    else:
        device_str = 'GPU X %s' % torch.cuda.device_count()

    print('''\nLoaded hyperparameters:
    Learning Rate:   %s
    Batch size:      %s
    Epochs:          %s
    Device:          %s
    Configuration:   %s
    ''' % (learning_rate, batch_size, max_epochs, device_str, exp_name))

    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            batch_size, config['geometry'])

    # Build the model
    net, loss_fn, optimizer, scheduler = build_model(
        config, device, train=True)

    if isinstance(loss_fn, Classification_Loss):
        loss_str = 'Alpha-balanced Classification Loss'
    elif isinstance(loss_fn, Focal_Loss):
        loss_str = 'Focal Loss'
    elif isinstance(loss_fn, Successive_E_F_Loss):
        loss_str = 'Successive Embedding and Focal Loss'
    elif isinstance(loss_fn, Successive_E_C_Loss):
        loss_str = 'Successive Embedding and Alpha-balanced Classification Loss'

    print('''\nBuilt model:
    Loss Function:   %s
    Optimizer:       %s
    Scheduler:       %s
    ''' % (loss_str, 'Adam', 'None'))

    # Tensorboard Logger
    train_writer = get_writer(config, 'train')

    # For picking up training at the epoch where you left off.
    # Edit this setting in config file.
    if config['resume_training']:
        start_epoch = config['resume_from']
        saved_ckpt_path = get_model_name(config, start_epoch)

        if config['mGPUs']:
            net.module.load_state_dict(
                torch.load(saved_ckpt_path, map_location=device))
        else:
            net.load_state_dict(
                torch.load(saved_ckpt_path, map_location=device))

        print('Successfully loaded trained checkpoint at {}'.format(
            saved_ckpt_path))
    else:
        start_epoch = 0

    # Do an initial validation round as a benchmark
    validation_round(net, device, exp_name, start_epoch)
    
    training_losses = []
    testing_losses = []

    # Train for max_epochs epochs
    for epoch in range(start_epoch, max_epochs):

        epoch_loss = 0
        net.train()

        with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch+1, max_epochs),
                  unit=' pointclouds') as progress:
            if epoch == 0:
                print('Classification Loss Only')
                loss_fn = Classification_Loss(config['classification_alpha'], config['reduction'])
            else:
                print('Successive Embedding and Classification')
                loss_fn = Successive_E_C_Loss(
                    device, lam=config['successive_lambda'],
                    alpha=config['classification_alpha'],
                    margin_s=config['embedding_margin_s'],
                    margin_d=config['embedding_margin_d'],
                    reduction=config['reduction'])            
            for input, label_map, instance_map, num_instances, image_id in \
                    train_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                optimizer.zero_grad()

                # Forward Prop
                predictions = net(input)
                
                if isinstance(loss_fn, Classification_Loss):
                    loss = loss_fn(predictions, label_map)
                    progress.set_postfix(
                        **{'loss': '{:.4f}'.format(abs(loss.item()))})

                else:
                    loss, embedding_loss, classification_loss = loss_fn(predictions, label_map)
                    # Update the progress bar this this batch's loss
                    progress.set_postfix(
                        **{'em/class': '{:.4f} {:.4f}'.format(abs(embedding_loss.item()), abs(classification_loss.item()))})

                # Back Prop
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Update the progress bar by moving it along by this batch size
                progress.update(input.shape[0])

        # Record Training Loss
        epoch_loss = epoch_loss / len(train_data_loader)
        training_losses.append(epoch_loss)
        train_writer.add_scalar('loss_epoch', epoch_loss, epoch + 1)
        print('Epoch {}: Training Loss: {:.5f}'.format(
            epoch + 1, epoch_loss))

        # Save Checkpoint
        if (epoch + 1) == max_epochs or \
                (epoch + 1) % config['save_every'] == 0:
            model_path = get_model_name(config, epoch + 1)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print('Checkpoint for epoch {} saved at {}\n'.format(
                epoch + 1, model_path))

        if scheduler is not None:
            scheduler.step()

        testing_loss = validation_round(net, device, exp_name, epoch)
        testing_losses.append(testing_loss)
        
        csvFilename = os.path.join(base_dir, 'experiments', exp_name, 'loss_log.csv')
    
        with open(csvFilename, 'w') as csvFile:
            csvWriter = csv.writer(csvFile)
            headers = ['Epoch Number', 'Training Loss', 'Testing Loss']
            csvWriter.writerow(headers)
            rows = []
            for i in range(0, len(training_losses)):
                rows.append([i, training_losses[i], testing_losses[i]])
            csvWriter.writerows(rows)

    print('Finished Training. Check the experiment folder for loss info, etc.')


def eval_loader(config, net, loss_fn, loader, loader_name, device):
    '''
    Evaluate the performance of the model on the specified data loader

    Parameters:
        config (dict): dict of hyperparam names and values for configuration
        net (torch.nn.Module): the network
        loss_fn (class): the loss function for the network
        loader (Dataset): a data loader in which to retrieve the input
            (batch_size should be 1)
        loader_name (str): the name to give the data loader (only for printing)
        device (torch.device): the device on which to run

    Returns:
        dict: dict of name-value pairs (useful information on the evaluation)
    '''
    net.eval()

    forward_pass_times = []
    losses = []

    with torch.no_grad():
        with tqdm(total=len(loader), desc='Evaluate %s Data: ' % loader_name,
                  unit=' pcl') as progress:
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
                progress.set_postfix(
                    **{'loss': '{:.4f}'.format(abs(loss.item()))})

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
        exp_name (str): the name of the config file to load
        device (torch.device): the device on which to run
        plot (bool): whether to plot the results for visualization
    '''
    # Load Hyperparameters
    config, _, _, _ = load_config(exp_name)

    # Build the model
    net, loss_fn = build_model(config, device, train=False)

    if isinstance(loss_fn, Classification_Loss):
        loss_str = 'Classification Loss'
    elif isinstance(loss_fn, Focal_Loss):
        loss_str = 'Focal Loss'
    elif isinstance(loss_fn, Successive_Loss):
        loss_str = 'Successive Embedding and Focal Loss'
    elif isinstance(loss_fn, Successive_E_C_Loss):
        loss_str = 'Successive Embedding and Alpha-balanced Classification Loss'

    print('''\nBuilt model:
    Loss Function:   %s
    Optimizer:       %s
    Scheduler:       %s
    ''' % (loss_str, 'Adam', 'None'))

    saved_ckpt_path = get_model_name(config)

    net.load_state_dict(
        torch.load(saved_ckpt_path, map_location=device))

    print('Successfully loaded trained checkpoint at {}\n'.format(
        saved_ckpt_path))

    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            1, config['geometry'])

    # Evaluate the performance on the training data
    metrics_train = eval_loader(
        config, net, loss_fn, train_data_loader, 'Train', device)

    # Evaluate the performance on the testing data
    metrics_val = eval_loader(
        config, net, loss_fn, test_data_loader, 'Test', device)

    total_inputs = len(train_data_loader) + len(test_data_loader)
    time_mean = (((metrics_train['time_mean'] * len(train_data_loader))
                 + (metrics_val['time_mean'] * len(test_data_loader)))
                 / total_inputs)

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

    '''.format(saved_ckpt_path, 'Classification Loss', time_mean,
               max(metrics_train['time_max'], metrics_val['time_max']),
               min(metrics_train['time_min'], metrics_val['time_min']),
               metrics_train['loss_mean'], metrics_train['loss_max'],
               metrics_train['loss_min'], metrics_val['loss_mean'],
               metrics_val['loss_max'], metrics_val['loss_min']))


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
    Test the network by using one image from the testing dataset,
    and save images of the process.

    Parameters:
        exp_name (str): the name of the config file to load
        device (torch.device): the device on which to run
        image_id (int): the image_id in the test data loader
    '''
    # Load Hyperparameters()
    config, _, _, _ = load_config(exp_name)

    # Build the model
    net, loss_fn = build_model(config, device, train=False)

    # Load the weights of the network
    net.load_state_dict(
        torch.load(get_model_name(config), map_location=device))

    if isinstance(loss_fn, Classification_Loss):
        loss_str = 'Classification Loss'
    elif isinstance(loss_fn, Focal_Loss):
        loss_str = 'Focal Loss'
    elif isinstance(loss_fn, Successive_Loss):
        loss_str = 'Successive Embedding and Focal Loss'
    elif isinstance(loss_fn, Successive_E_C_Loss):
        loss_str = 'Successive Embedding and Alpha-balanced Classification Loss'

    print('''\nBuilt model:
    Loss Function:   %s
    Weights:         %s
    ''' % (loss_str, get_model_name(config)))

    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            config['batch_size'], config['geometry'])

    net.eval()

    with torch.no_grad():
        input, label_map, prediction, loss, time = eval_one(
            net, loss_fn, test_data_loader, image_id, device)

    truth_filename = 'EVAL_num_%s_truth.jpg' % image_id
    pcl_filename = 'EVAL_num_%s_point_cloud.jpg' % image_id
    prediction_filename = 'EVAL_num_%s_unet_output.jpg' % image_id

    save_images(exp_name, input, label_map[0], prediction[0],
                pcl_filename, truth_filename, prediction_filename)
    save_dir = os.path.join(base_dir, 'experiments', exp_name, 'images')

    print('''\nEvaluated Image %s:
    Forward Pass Time:   %s
    Loss:                %s
    Ground Truth Image   %s
    Point Cloud Image    %s
    U-Net Output Image   %s
    ''' % (image_id, time, loss,
           os.path.join(save_dir, truth_filename),
           os.path.join(save_dir, pcl_filename),
           os.path.join(save_dir, prediction_filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net training module')
    parser.add_argument(
        'mode', choices=['train', 'eval', 'test'], help='mode for the model')
    parser.add_argument(
        '--test_id', type=int, default=25, help='id of the image to test')
    args = parser.parse_args()

    # Choose a device for the model
    if torch.cuda.is_available():
        if not args.mode == 'train':
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.mode == 'train':
        train(exp_name, device)
    if args.mode == 'eval':
        evaluate_model(exp_name, device, plot=False)
    if args.mode == 'test':
        test(exp_name, device, image_id=args.test_id)
