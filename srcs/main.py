import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import csv
import random

import cv2 as cv
from tqdm import tqdm
from matplotlib import cm

from config import base_dir, exp_name
from dataset import get_data_loader
from loss.classification_loss import Classification_Loss
from loss.focal_loss import Focal_Loss
from loss.successive_e_c_loss import Successive_E_C_Loss
from loss.successive_e_f_loss import Successive_E_F_Loss
from loss.discriminative_loss import Discriminative_Loss
from utils import get_model_path, load_config, mkdir_p, get_loss_string

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
    net = UNet(config['geometry'], output_dim=1,
               use_batchnorm=config['use_bn'])

    # Determine the loss function to be used and initialize it
    if config['training_loss'] == 'c' or not train:
        loss_fn = Classification_Loss(
            config['classification_alpha'], config['reduction'])
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
    elif config['training_loss'] == 'd':
        loss_fn = Discriminative_Loss()
    else:
        raise ValueError('loss argument must be in [c, f, d, s_ef, s_ec]')

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


def save_images(exp_name, epoch, input, label_map, prediction,
                pcl_filename, truth_filename, pred_filename,
                em_filename):
    '''
    Save images of the input Point Cloud,
    the ground truth, and the U-Net prediction

    Parameters:
        exp_name (str): the name of the config file to load
        epoch (any): epoch number or indicator
        input (Tensor): input to the network
        label_map (Tensor): labels, retrieved from a data loader
        prediction (Tensor): output from the network
        pcl_filename (string): file in which to save the point cloud
        truth_filename (string): file in which to save the ground truth
        pred_filename (string): file in which to save the prediction
        em_filename (string): file in which to save the embedding,
            or None if you wish not to save the embedding
    '''
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)

    # If not already created, make a directory to save the images
    save_dir = os.path.join(base_dir, 'experiments', exp_name,
                            'images_{}epoch'.format(epoch))
    mkdir_p(save_dir)

    # Create an array that maps instance number to color for visualization
    color_map = cm.get_cmap('jet')
    colors = [color_map(x) for x in np.arange(
        0, 1, 1./config['geometry']['max_num_instances'])]
    for i, y in enumerate(colors):
        # Disregard alpha value
        colors[i] = [tuple(255 * x for x in y)[0:-1]]

    # Copy to a Tensor for use in GPU tasks
    colors_arr = torch.Tensor(label_map.shape[0], 3)
    for i, x in enumerate(colors):
        colors_arr[i][0] = x[0][0]
        colors_arr[i][1] = x[0][1]
        colors_arr[i][2] = x[0][2]
    colors_arr = colors_arr.type(torch.uint8)

    # Ground Truth Image
    if label_map is not None:
        if label_map.shape[0] == 1:
            # CxWxH to WxHxC & convert to grayscale image format
            truth = np.array(
                label_map.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
            image_truth = cv.cvtColor(truth, cv.COLOR_GRAY2BGR)
        else:
            image_truth = np.zeros(
                (config['geometry']['input_shape'][0],
                 config['geometry']['input_shape'][1], 3), np.uint8)
            points = torch.nonzero(label_map)
            for p in points:
                image_truth[np.int64(p[1]), np.int64(p[2])] = colors[p[0]][0]
        cv.imwrite(os.path.join(save_dir, truth_filename), image_truth)

    # Point Cloud Image
    if input is not None:
        image_pcl = np.zeros(
            (config['geometry']['input_shape'][0],
             config['geometry']['input_shape'][1], 3), np.uint8)
        points = torch.nonzero(input)
        for p in points:
            # Color gradient based on height
            r, g, b, a = color_map(
                np.float(p[0]) / config['geometry']['input_shape'][2])
            image_pcl[np.int64(p[1]), np.int64(p[2])] = \
                (r * 255, g * 255, b * 255)
        cv.imwrite(os.path.join(save_dir, pcl_filename), image_pcl)

    # Prediction Image
    if prediction is not None:
        prediction = torch.sigmoid(prediction)
        if prediction.shape[0] == 1:
            prediction_gray = np.array(
                prediction.cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
            cv.imwrite(os.path.join(save_dir, pred_filename), prediction_gray)
        else:
            prediction = prediction.contiguous().view(
                prediction.shape[0], prediction.shape[1] *
                prediction.shape[2]).permute(1, 0)
            image_pred = torch.zeros((
                config['geometry']['input_shape'][0]
                * config['geometry']['input_shape'][1], 3))
            one_hot = torch.argmax(prediction, dim=1)
            vals, _ = torch.max(prediction, dim=1)
            image_pred = torch.index_select(colors_arr.cuda(), 0, one_hot)
            maxes = torch.zeros_like(image_pred).type(torch.float32)
            maxes[:, 0] = vals
            maxes[:, 1] = vals
            maxes[:, 2] = vals
            image_pred = torch.mul(image_pred, maxes).reshape(
                (config['geometry']['input_shape'][0],
                 config['geometry']['input_shape'][1], 3))
            image_pred = image_pred.cpu().numpy()
            cv.imwrite(os.path.join(save_dir, pred_filename), image_pred)

    # Embedding Image
    if em_filename is not None:
        # Flatten labels and predictions into vectors
        flattened_labels = label_map.view(-1, 1)
        flattened_preds = prediction.view(-1, 1)

        # RANSAC a certain number of negative and positive pixels
        pos_pixels = []
        neg_pixels = []
        total_pixels = 250
        while len(pos_pixels) + len(neg_pixels) < total_pixels:
            index = int(random.random() * len(flattened_preds))
            if flattened_labels[index] == 1 and \
                    len(pos_pixels) < total_pixels / 2:
                pos_pixels.append(flattened_preds[index])
            elif flattened_labels[index] == 0 and \
                    len(neg_pixels) < total_pixels / 2:
                neg_pixels.append(flattened_preds[index])

        # Determine the Y axis scale and values
        max_y = max(max(pos_pixels), max(neg_pixels))
        min_y = min(min(pos_pixels), min(neg_pixels))
        y_scale = 400 / (abs(max_y - min_y))

        # Create the image, plotting all the pixels
        image_embedding = np.zeros(
            (config['geometry']['input_shape'][0],
             config['geometry']['input_shape'][1], 3), np.uint8)
        for y_val in pos_pixels:
            x_image = np.random.normal(
                loc=config['geometry']['input_shape'][0] / 2, scale=60)
            x_image = int(
                max(1, min(x_image, config['geometry']['input_shape'][0] - 1)))
            y_image = int((y_val - min_y) * y_scale)
            image_embedding = cv.circle(
                image_embedding, (x_image, y_image),
                radius=1, color=(0, 0, 255), thickness=-1)
        for y_val in neg_pixels:
            x_image = np.random.normal(
                loc=config['geometry']['input_shape'][0] / 2, scale=60)
            x_image = int(
                max(1, min(x_image, config['geometry']['input_shape'][0] - 1)))
            y_image = int((y_val - min_y) * y_scale)
            image_embedding = cv.circle(
                image_embedding, (x_image, y_image),
                radius=1, color=(255, 0, 0), thickness=-1)
        cv.imwrite(os.path.join(save_dir, em_filename), image_embedding)


def validation_round(net, device, exp_name, epoch_num, save_embedding=False):
    '''
    Find testing data loss and save images of the pipeline if applicable

    Parameters:
        net (torch.nn.Module): the network
        device (torch.device): the device on which to run
        exp_name (str): the name of the config file to load
        epoch_num (int): the epoch number (for saving images)
        save_embedding (bool): whether to save an image of the embedding
    '''
    net.eval()

    # Load hyperparameters
    config, _, _, _ = load_config(exp_name)
    batch_size = config['validation_batch_size']

    # Always use Classification loss for evaluation
    loss_fn = Classification_Loss(
        config['classification_alpha'], config['reduction'])

    # Retrieve the datasets for training and testing
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
        batch_size=batch_size, geometry=config['geometry'])

    with torch.no_grad():
        # This will keep track of total average loss for all test data
        ave_loss = 0

        with tqdm(total=num_val, desc='Validation: ', unit=' pointclouds',
                  leave=False, colour='magenta') as progress:
            for input, label_map, instance_map, num_instances, image_id in \
                    test_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                # Forward Prop
                predictions, instance_predictions = net(input)

                loss = loss_fn(predictions, label_map)

                # Update the progress bar with the current batch loss
                progress.set_postfix(
                    **{'batch loss': '{:.4f}'.format(abs(loss.item()))})
                ave_loss += abs(loss.item())

                # After some epochs, save an image of the
                # input, output, and ground truth for visualization
                if config['visualize']:
                    if epoch_num + 1 in config['vis_after_epoch'] or \
                            config['vis_every_epoch']:
                        truth_filename = 'epoch_%s_image_%s_truth.jpg' % \
                            (epoch_num, progress.n)
                        pcl_filename = 'epoch_%s_image_%s_point_cloud.jpg' % \
                            (epoch_num, progress.n)
                        prediction_filename = 'epoch_%s_image_%s_unet.jpg' % \
                            (epoch_num, progress.n)
                        em_filename = 'epoch_%s_image_%s_embedding.jpg' % \
                            (epoch_num, progress.n)

                        save_images(exp_name, epoch_num, input[0],
                                    instance_map[0], instance_predictions[0],
                                    pcl_filename, truth_filename,
                                    prediction_filename,
                                    em_filename if save_embedding else None)

                # Update the progress bar, moving it along by one batch size
                progress.update(input.shape[0])

        ave_loss = ave_loss / len(test_data_loader)
        if epoch_num == 0:
            print('Initial Benchmark Validation Loss: {:.5f}'.format(ave_loss))
        else:
            print('Validation Loss After Epoch {} {:.5f}'.format(
                epoch_num, ave_loss))
        return ave_loss


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

    print('''\nBuilt model:
    Loss Function:   %s
    Optimizer:       %s
    Scheduler:       %s
    ''' % (get_loss_string(loss_fn), 'Adam', 'None'))

    # For picking up training at the epoch where you left off.
    # Edit this setting in config file.
    if config['resume_training']:
        start_epoch = config['resume_from']
        saved_ckpt_path = get_model_path(config, start_epoch)

        if config['mGPUs']:
            net.module.load_state_dict(
                torch.load(saved_ckpt_path, map_location=device))
        else:
            net.load_state_dict(
                torch.load(saved_ckpt_path, map_location=device))

        print('Successfully loaded trained checkpoint at {}'.format(
            saved_ckpt_path))
    else:
        start_epoch = 1

    # Do an initial validation round as a benchmark
    validation_round(net, device, exp_name, start_epoch - 1,
                     save_embedding=config['save_embedding'])

    training_losses = []
    testing_losses = []

    for epoch in range(start_epoch, max_epochs):

        epoch_loss = 0
        net.train()

        with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch, max_epochs),
                  unit=' pointclouds', leave=False, colour='green') \
                as progress:
            for input, label_map, instance_map, num_instances, image_id in \
                    train_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                optimizer.zero_grad()

                # Forward Prop
                predictions, instance_predictions = net(input)

                if isinstance(loss_fn, Successive_E_F_Loss) or \
                        isinstance(loss_fn, Successive_E_C_Loss):
                    loss, em_loss, class_loss = loss_fn(predictions, label_map)
                    progress.set_postfix(
                        **{'em/class': '{:.4f} {:.4f}'.format(
                            abs(em_loss.item()), abs(class_loss.item()))})
                else:
                    loss = loss_fn(predictions, label_map)
                    progress.set_postfix(
                        **{'loss': '{:.4f}'.format(abs(loss.item()))})

                # Back Prop
                loss.backward()
                optimizer.step()

                # Update the progress bar by moving it along by this batch size
                progress.update(input.shape[0])

        # Record Training Loss
        epoch_loss = epoch_loss / len(train_data_loader)
        training_losses.append(epoch_loss)
        print('Epoch {}: Training Loss: {:.5f}'.format(epoch, epoch_loss))

        # Save Checkpoint
        if epoch == max_epochs or \
                epoch % config['save_every'] == 0:
            model_path = get_model_path(config, epoch)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print('Checkpoint for epoch {} saved at {}\n'.format(
                epoch, model_path))

        if scheduler is not None:
            scheduler.step()

        testing_losses.append(validation_round(
            net, device, exp_name, epoch,
            save_embedding=config['save_embedding']))

        csvFilename = os.path.join(
            base_dir, 'experiments', exp_name, 'loss_log.csv')

        # Every epoch, re-write the csv file containing the losses
        with open(csvFilename, 'w') as csvFile:
            csvWriter = csv.writer(csvFile)
            headers = ['Epoch Number', 'Training Loss', 'Testing Loss']
            csvWriter.writerow(headers)
            rows = []
            for i in range(0, len(training_losses)):
                rows.append([i, training_losses[i], testing_losses[i]])
            csvWriter.writerows(rows)

    print('Finished Training. Check this folder for loss info, images, etc.:')
    print(os.path.join(base_dir, 'experiments', exp_name))


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


def evaluate_model(exp_name, device):
    '''
    Determine the total performance of the network on all data

    Parameters:
        exp_name (str): the name of the config file to load
        device (torch.device): the device on which to run
    '''
    # Load Hyperparameters
    config, _, _, _ = load_config(exp_name)

    # Build the model
    net, loss_fn = build_model(config, device, train=False)

    print('''\nBuilt model:
    Loss Function:   %s
    Optimizer:       %s
    Scheduler:       %s
    ''' % (get_loss_string(loss_fn), 'Adam', 'None'))

    saved_ckpt_path = get_model_path(config)

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
        torch.load(get_model_path(config), map_location=device))

    print('''\nBuilt model:
    Loss Function:   %s
    Weights:         %s
    ''' % (get_loss_string(loss_fn), get_model_path(config)))

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

    save_images(exp_name, 'final', input, label_map[0], prediction[0],
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
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.mode == 'train':
        train(exp_name, device)
    if args.mode == 'eval':
        evaluate_model(exp_name, device)
    if args.mode == 'test':
        test(exp_name, device, image_id=args.test_id)
