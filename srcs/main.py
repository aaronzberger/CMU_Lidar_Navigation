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

from loss import ClassificationLoss, EmbeddingLoss
from dataset import get_data_loader
from unet import UNet
from utils import get_model_name, load_config, get_writer, plot_pr_curve
from postprocess import compute_line_matches, compute_ap, extract_lines, compute_precision_recall
from torchvision.utils import make_grid
from shapely.geometry import LineString
from tqdm import tqdm

from sklearn.decomposition import PCA
from bev import BEV

def build_model(config, device, output="class", train=True):
    if output == "class":
        out_channels = 1
    else:
        out_channels = config["embedding_dim"]
        
    net = UNet(config['geometry'], use_batchnorm=config['use_bn'], output_dim=out_channels,
            feature_scale=config["layer_width_scale"])

#     Determine the loss function to be used
    if output == "class":
        loss_fn = ClassificationLoss(device, config, num_classes=1)
    elif output == "embedding":
        loss_fn = EmbeddingLoss(device, config, embedding_dim=config['embedding_dim'])

#     Determine whether to run on multiple GPUs
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


def distance(x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    return math.sqrt(dx**2 + dy**2)

def angle_to_line(l1):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    
    a = (dy*(200-y1)+dx*(200-x1))/det
    closestX, closestY = x1+a*dx, y1+a*dy
    
    return math.atan2(closestY - 200, closestX - 200)

def line_to_point(x, y, a, b, c):
    return (abs(a * x + b * y + c)) / (math.sqrt(a * a + b * b))

def line_to_line(l1, l2):
    line1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
    line2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])
    
    return line1.distance(line2)
    
def segment_to_standard(l1):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    a = y1 - y2
    b = x2 - x1
    c = (x1-x2)*y1 + (y2-y1)*x1
    
    return a, b, c

def intersect(l1, l2):
    line1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
    line2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])
    
    return line1.intersects(line2)

def lines_are_close(l1, l2, dist, theta_deg):
    a1, b1, c1 = segment_to_standard(l1)
    line1Dist = line_to_point(200, 200, a1, b1, c1)
    line1Angle = angle_to_line(l1)
    
    a2, b2, c2 = segment_to_standard(l2)
    line2Dist = line_to_point(200, 200, a2, b2, c2)
    line2Angle = angle_to_line(l2)
    
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    x3, y3 = l2[0][0], l2[0][1]
    x4, y4 = l2[0][2], l2[0][3]
    
    shortestDistance = line_to_point(x1, y1, a2, b2, c2)
    shortestDistance = line_to_point(x2, y2, a2, b2, c2) if line_to_point(x2, y2, a2, b2, c2) <= shortestDistance else shortestDistance
    shortestDistance = line_to_point(x3, y3, a1, b1, c1) if line_to_point(x3, y3, a1, b1, c1) <= shortestDistance else shortestDistance
    shortestDistance = line_to_point(x4, y4, a1, b1, c1) if line_to_point(x4, y4, a1, b1, c1) <= shortestDistance else shortestDistance

    return intersect(l1, l2) or shortestDistance < dist or (abs(line1Dist - line2Dist) < dist and abs(line1Angle - line2Angle) < math.radians(theta_deg))


# Arugument pred_map should be the simgoid of the output of the UNet model
def line_score(l1, pred_map):    
    # Constants
    layer2Weight = 0.5
    layer3Weight = 0.2
    lengthWeight = 1.5
    
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    layer1Score = 0
    layer2Score = 0
    layer3Score = 0
    
    # This is the number of points for each layer
    layer1Points = 0
    layer2Points = 0
    layer3Points = 0

    # Bresenham Algorithm for drawing lines
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    countInX = True if dx > dy else False

    while not (x1 == x2 and y1 == y2):        
        # Score the current pixel
        layer1Score += pred_map[y1, x1] - 5
        layer1Points += 1

        if countInX:
            if x1 <= 398 and x1 >= 2:
                layer2Score += pred_map[y1, x1+1]
                layer2Score += pred_map[y1, x1-1]
                layer2Points += 2
            if x1 <= 397 and x1 >= 3:
                layer3Score += pred_map[y1, x1+2]
                layer3Score += pred_map[y1, x1-2]
                layer3Points += 2
        else:
            if y1 <= 398 and y1 >= 2:
                layer2Score += pred_map[y1+1, x1]
                layer2Score += pred_map[y1-1, x1]
                layer2Points += 2
            if y1 <= 397 and y1 >= 3:
                layer3Score += pred_map[y1+2, x1]
                layer3Score += pred_map[y1-2, x1]
                layer3Points += 2

        # Move to the next pixel
        e2 = err << 1
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    finalLayer1 = layer1Score / float(layer1Points)
    finalLayer2 = layer2Score / float(layer2Points)
    finalLayer3 = layer3Score / float(layer3Points)
    
    final_score = finalLayer1 + (finalLayer2 * layer2Weight) + (finalLayer3 * layer3Weight)
    
    line_length = math.sqrt(dx ** 2 + dy ** 2)
    final_score += (line_length * lengthWeight)

    return final_score

    
def distance_to_origin(l1):
    a, b, c = segment_to_standard(l1)
    return line_to_point(200, 200, a, b, c)


def validation_round(net, loss_fn, device, exp_name, round_num):
    net.eval()
    
    # Load hyperparameters
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)

    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
        batch_size, config['geometry'])
    
    with torch.no_grad():
        val_loss = 0
        image_number = 0
        
        with tqdm(total=num_val, desc='Validation: ', unit=' pointclouds') as progress:
            for input, label_map, _, _, _ in test_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                predictions = net(input)

                loss = loss_fn(predictions, label_map)
                
                progress.set_postfix(**{'loss (batch)': abs(loss.item())})
                val_loss += abs(loss.item())
                
                # To better visualize the images, exagerate the difference between 0 and 1
                predictions = torch.sigmoid(predictions)

                if config['visualize']:
                    for num in config['vis_after_epoch']:
                        if round_num == num:
                            # Save the ground truth image
                            # With the first ground truth data in the batch, convert channel order: 
                            # CxWxH to WxHxC and convert to grayscale image format (0-255 and 8-bit int)
                            truth = np.array(label_map[0].cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
                            image_truth = cv.cvtColor(truth, cv.COLOR_GRAY2BGR)                            
                            
                            # Save the prediction image
                            prediction_gray = np.array(predictions[0].cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
                            prediction_bgr = cv.cvtColor(prediction_gray, cv.COLOR_GRAY2BGR)
                            filename = "images/%s_unet_output.jpg" % image_number
                            cv.imwrite(filename, prediction_bgr)
        
                            # Construct and save an image of the point cloud
#                             pcl = np.array(input[0].cpu() * 255, dtype=np.uint8)
#                             image_pcl = np.amax(pcl, axis=0)
#                             filename = "images/%s_pcl.jpg" % image_number
#                             cv.imwrite(filename, image_pcl)

                            # Blur the image so the lines are more defined
                            kernel = np.ones((4, 4), np.float32)
                            blurred = cv.filter2D(prediction_gray, -1, kernel)
                            filename = "images/%s_blurred.jpg" % image_number
                            cv.imwrite(filename, blurred)
                            
                            # Find the lines
                            lines = cv.HoughLinesP(blurred, 2, (np.pi / 180), 150, None, 25, 50)
                                                            #  min_intersections, None, min_points, max_gap
                            if lines is None:
                                continue

                            # LINE CLUSTERING
                            # Determine whether each line is not yet clustered (1 or 0)
                            tracker = np.ones(len(lines))

                            # For each line, see which other lines it matches and put those in a group (O(n) = n^2)
                            newLines = []
                            for i in range(0, len(lines)):
                                if tracker[i] == 1:
                                    group = []
                                    group.append(lines[i])
                                    for j in range(1, len(lines)):
                                        if tracker[j] == 1:
                                        # A line must be similar to any other line in the group. 
                                        # This could be changed for efficiency (ie: always keep the longest or average 
                                        # line as a comparison for new lines)
                                            for line in group:
                                                if lines_are_close(line, lines[j], 20, 10):
                                                    group.append(lines[j])
                                                    tracker[j] = 0
                                                    break
                                    tracker[i] = 0
                                    newLines.append(group)
                            
                            # Make an image containing all detected lines, color coded by cluster
                            color_options = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127, 0, 0), (0, 127, 0), (0, 0, 127), (127, 127, 0), (127, 0, 127), (0, 127, 127), (0, 0, 127)]
                            color_clusters = np.copy(prediction_bgr)
                            for i in range(0, len(newLines)):
                                for line in newLines[i]:
                                    cv.line(color_clusters, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color_options[i], 1, cv.LINE_AA)
                            filename = "images/%s_color_clusters.jpg" % image_number
                            cv.imwrite(filename, color_clusters)

                            # Choose the best line in each cluster to keep (uses the scoring function above)
                            finalLines = []
                            if len(newLines) > 0:
                                for group in newLines:
                                    bestLine = group[0]
                                    bestScore = line_score(group[0], blurred)
                                    for i in range(1, len(group)):
                                        lineScore = line_score(group[i], blurred)
                                        if lineScore >= bestScore:
                                            bestLine = group[i]
                                            bestScore = lineScore
                                    finalLines.append(bestLine)
                                    
                            # Make and save an image of the final line predictions
                            line_prediction = np.copy(cv.cvtColor(blurred, cv.COLOR_GRAY2BGR))
                            for i in range(0, len(finalLines)):
                                l = finalLines[i][0]
                                cv.line(line_prediction, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)
                                filename = "images/%s_final.jpg" % image_number
                                cv.imwrite(filename, line_prediction)
                            
                image_number = image_number + 1
                
                # Update the tqdm progress bar, moving it along by one batch size
                progress.update(input.shape[0])
                
        val_loss = val_loss / len(test_data_loader)
        print("Validation Round Loss: %s" % val_loss)
                

def eval_batch(config, net, loss_fn, loader, device, eval_range='all'):
    # net.eval()

    # if config['mGPUs']:
    #     net.module.set_decode(True)
    # else:
    #     net.set_decode(True)
    
    running_loss = 0
    all_scores = []
    all_matches = []
    log_images = []
    gts = 0
    preds = 0
    t_fwd = 0
    t_pp = 0

    log_img_list = random.sample(range(len(loader.dataset)), 10)

    with torch.no_grad():
        for i, data in enumerate(loader):
            #print("batch", i)
            tic = time.time()

            input, label_map, _, _, image_id = data
            input = input.to(device)

            label_map = label_map.to(device)
            tac = time.time()
            predictions = net(input)
            predictions = torch.sigmoid(predictions)

            t_fwd += time.time() - tac

            #print('did predictions')

            loss = loss_fn(predictions, label_map)
            running_loss += loss
            t_fwd += (time.time() - tic)
            
            toc = time.time()
            
            # Parallel post-processing
            predictions = [t.squeeze().numpy() for t in torch.split(predictions.cpu(), 1, dim=0)]
            batch_size = len(predictions)

            #print([arr.shape for arr in predictions])

            #print('start pool')
            with Pool (processes=3) as pool:
                # TODO FIXME add parameters for line extraction to config
                # and pass config instead of kwarg parameters
                lines = pool.starmap(
                    extract_lines, [(pred,) for pred in predictions])

            print([len(l) for l in lines])
            t_pp += time.time() - toc

            #print("done postprocess")

            args = []
            for j in range(batch_size):
                label_list = loader.dataset.get_labels(image_id[j].item())

                gts += len(label_list)
                preds += len(lines[j])

                # DISABLE VISUALIZATION
                # if image_id[j] in log_img_list:
                #     input_np = input[j].cpu().permute(1, 2, 0).numpy()
                #     pred_image = get_bev(input_np, corners)
                #     log_images.append(pred_image)

                arg = (np.array(label_list), lines[j])
                args.append(arg)

            # Parallel compute matches
            with Pool (processes=3) as pool:
                matches = pool.starmap(compute_line_matches, args)
            
            for j in range(batch_size):
                pred_matches_j = list(matches[j][1])
                all_matches.extend(pred_matches_j)
                
            
    all_matches = np.array(all_matches)

    metrics = {}
    precision, recall, tp, fp, fn = compute_precision_recall(all_matches, gts, preds)
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['FP'] = fp
    metrics['FN'] = fn
    metrics['TP'] = tp

    metrics['Forward Pass Time'] = t_fwd/len(loader.dataset)
    metrics['Postprocess Time'] = t_pp/len(loader.dataset) 

    running_loss = running_loss / len(loader)
    metrics['loss'] = running_loss

    return metrics, precision, recall, log_images


def eval_dataset(config, net, loss_fn, loader, device, e_range='all'):
    net.eval()
    # if config['mGPUs']:
    #     net.module.set_decode(True)
    # else:
    #     net.set_decode(True)

    t_fwds = 0
    t_post = 0
    loss_sum = 0

    img_list = range(len(loader.dataset))
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    log_img_list = random.sample(img_list, 10)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    log_images = []

    with torch.no_grad():
        for image_id in img_list:
            #tic = time.time()
            num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
                eval_one(net, loss_fn, config, loader, image_id, device, plot=False)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))

            t_fwds += t_forward
            t_post += t_nms

            if image_id in log_img_list:
                log_images.append(pred_image)
            #print(time.time() - tic)
            
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['loss'] = loss_sum / len(img_list)
    metrics['Forward Pass Time'] = t_fwds / len(img_list)
    metrics['Postprocess Time'] = t_post / len(img_list)

    return metrics, precisions, recalls, log_images


def train(exp_name, device, output):
    # Load Hyperparameters
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)
    
    print('''\nLoaded hyperparameters:
    Learning Rate:   %s
    Batch size:      %s
    Epochs:          %s
    Device:          %s
    ''' % (learning_rate, batch_size, max_epochs, device.type if device.type == 'cpu' else "GPU X %s" % torch.cuda.device_count()))
    
    # Dataset and DataLoader
    train_data_loader, test_data_loader, num_train, num_val = get_data_loader(
            batch_size, config['geometry'])

    # Model
    net, loss_fn, optimizer, scheduler = build_model(config, device, output, train=True)
    
    print('''\nBuilt model:
    Loss Function:   %s
    Optimizer:       %s
    Scheduler:       %s
    ''' % ("Classification Loss" if output == "class" else "Embedding Loss", "Adam", "None"))

    # Tensorboard Logger
    train_writer = get_writer(config, 'train')
    val_writer = get_writer(config, 'val')

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
    running_loss = 0

    # Do an initial validation as a benchmark
    validation_round(net, loss_fn, device, exp_name, start_epoch)

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

                # Forward
                predictions = net(input)
                if output == "class":
                    loss = loss_fn(predictions, label_map, mask)
                elif output == "embedding":
                    loss = loss_fn(predictions, instance_map, num_instances)

                progress.set_postfix(**{'loss (batch)': loss.item()})

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_loss += loss.item()

                if step % config['log_every'] == 0:
                    running_loss = running_loss / config['log_every']
                    train_writer.add_scalar('running_loss', running_loss, step)
                    running_loss = 0

                step += 1
                
                progress.update(input.shape[0])

        # Record Training Loss
        epoch_loss = epoch_loss / len(train_data_loader)
        train_writer.add_scalar('loss_epoch', epoch_loss, epoch + 1)
        print("Epoch {}: Training Loss: {:.5f}".format(
            epoch + 1, epoch_loss))

        # Run Validation
        # TODO FIXME Skip validation for now
#         validation_round(net, loss_fn, device, exp_name, epoch+1)
        
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
            print("Checkpoint saved at {}\n".format(model_path))

        if scheduler is not None:
            scheduler.step()

    print('Finished Training')


def eval_one(net, loss_fn, config, loader, image_id, device, plot=False, verbose=False):
    input, label_map, image_id = loader.dataset[image_id]
    input = input.to(device)
    label_map, label_list = loader.dataset.get_label(image_id)
    loader.dataset.reg_target_normalize(label_map)
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)

    # Forward Pass
    t_start = time.time()
    pred = net(input.unsqueeze(0))
    t_forward = time.time() - t_start

    loss, running_loss, loc_loss = loss_fn(pred, label_map)
    pred.squeeze_(0)
    cls_pred = pred[0, ...]

    if verbose:
        print("Forward pass time", t_forward)


    # Filter Predictions
    t_start = time.time()
    corners, scores = filter_pred(config, pred)
    t_post = time.time() - t_start

    if verbose:
        print("Non max suppression time:", t_post)

    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(scores)
    input_np = input.cpu().permute(1, 2, 0).numpy()
    pred_image = get_bev(input_np, corners)

    # if plot == True:
        # Visualization
        # plot_bev(input_np, label_list, window_name='GT')
        # plot_bev(input_np, corners, window_name='Prediction')
        # plot_label_map(cls_pred.numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item(), t_forward, t_post


def experiment(exp_name, device, eval_range='all', plot=True):
    config, _, _, _ = load_config(exp_name)
    net, loss_fn = build_model(config, device, train=False)

    path = get_model_name(config)
    state_dict = torch.load(path, map_location=device)

    print("Loaded checkpoint from " + path)

    #if config['mGPUs']:
    #    net.module.load_state_dict(state_dict)
    #else:
    #    net.load_state_dict(state_dict)

    train_loader, val_loader = get_data_loader(
            config['batch_size'], config['geometry'],
    )

    #Train Set
    #train_metrics, train_precisions, train_recalls, _ = eval_batch(config, net, loss_fn, train_loader, device, eval_range)
    # print("Training mAP", train_metrics['AP'])
    #fig_name = "PRCurve_train_" + config['name']
    #legend = "AP={:.1%}".format(train_metrics['AP'])
    #plot_pr_curve(train_precisions, train_recalls, legend, name=fig_name)

    # Val Set
    val_metrics, val_precision, val_recall, _ = eval_batch(config, net, loss_fn, val_loader, device, eval_range)

    print("Validation precision", val_metrics['Precision'])
    print("Validation recall", val_metrics['Recall'])
    print("Validation FP", val_metrics['FP'])
    print("Validation FN", val_metrics['FN'])
    print("Validation TP", val_metrics['FN'])
    print("Net Fwd Pass Time on average {:.4f}s".format(val_metrics['Forward Pass Time']))
    print("Postprocess Time on average {:.4f}s".format(val_metrics['Postprocess Time']))

    #fig_name = "PRCurve_val_" + config['name']
    #legend = "AP={:.1%}".format(val_metrics['AP'])
    #plot_pr_curve(val_precisions, val_recalls, legend, name=fig_name)


def test(exp_name, device, image_id):
    config, _, _, _ = load_config(exp_name)
    net, loss_fn = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config), map_location=device))
    # net.load_state_dict(torch.load("experiments/default/config.json", map_location=device))

    net.set_decode(True)
    train_loader, val_loader = get_data_loader(1, config['geometry'])
    net.eval()

    with torch.no_grad():
        num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
            eval_one(net, loss_fn, config, train_loader, image_id, device, plot=True)

        TP = (pred_match != -1).sum()
        print("Loss: {:.4f}".format(loss))
        print("Precision: {:.2f}".format(TP/num_pred))
        print("Recall: {:.2f}".format(TP/num_gt))
        print("forward pass time {:.3f}s".format(t_forward))
        print("nms time {:.3f}s".format(t_nms))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument('mode', choices=['train', 'val', 'test'], help='name of the experiment')
    parser.add_argument('--name', required=True, help="name of the experiment")
    parser.add_argument('--eval_range', type=int, help="range of evaluation")
    parser.add_argument('--test_id', type=int, default=25, help="id of the image to test")
    parser.add_argument('--output', required=True, help="output of the model")
    args = parser.parse_args()

    if args.output not in ["class", "embedding"]:
        raise ValueError("output must be one of {class, embedding}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   

    if args.mode=='train':
        train(args.name, device, args.output)
    if args.mode=='val':
        if args.eval_range is None:
            args.eval_range='all'
        experiment(args.name, device, eval_range=args.eval_range, plot=False)
    if args.mode=='test':
        test(args.name, device, image_id=args.test_id)

    # before launching the program! CUDA_VISIBLE_DEVICES=0, 1 python main.py .......
