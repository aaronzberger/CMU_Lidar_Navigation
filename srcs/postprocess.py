'''
Non Max Suppression
IOU, Recall, Precision, Find overlap and Average Precisions
Source Code is adapted from github.com/matterport/MaskRCNN

'''

import numpy as np
import torch
from shapely.geometry import Polygon
import cv2

from fit_line import fit_line

def extract_lines(class_map, pred_thresh=0.8, min_ratio=10):
    threshed = class_map > pred_thresh

    _, contours, _ = cv2.findContours(
        threshed.astype('uint8'), 
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [
        c for c in contours 
        if cv2.contourArea(c) > 1
    ]

    height, width = threshed.shape 
    x = np.arange(width)
    y = np.arange(height)

    coords = np.meshgrid(x, y, indexing='ij')
    coords = np.stack(coords, axis=2).reshape(-1, 2).astype('float')

    lines = []
    for c in contours:
        mask_img = np.full((height, width, 3), 0, dtype='uint8')
        cv2.drawContours(mask_img, [c], 0, (255, 255, 255),
                thickness=-1)
        mask = (mask_img[:,:,0] > 0).reshape(-1)

        row_coords = coords[mask]

        a, b, d, ratio = fit_line(row_coords)
        if ratio < min_ratio:
            continue

        mean = np.mean(row_coords, axis=0)

        n_p = np.array([b, -a], dtype='float')
        row_coords_centered = row_coords - mean
        x = row_coords_centered * n_p

        proj = x.sum(axis=1)

        max_np = proj.max()
        min_np = proj.min()

        p1 = mean + (n_p * max_np)
        p2 = mean + (n_p * min_np)

        lines.append(np.concatenate((p1, p2)))

    if len(lines) > 0:
        lines = np.stack(lines, axis=0)

    return lines

def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_line_distance(line, lines1):
    x1, y1, x2, y2 = line
    
    lines2 = np.concatenate((lines1[:,2:4], lines1[:,0:2]), axis=1)
 
    # Frechet distance assuming 1 -> 2
    dist11 = np.linalg.norm(line[0:2] - lines1[:,0:2], axis=1)
    dist12 = np.linalg.norm(line[2:4] - lines1[:,2:4], axis=1)

    #print("dist11", dist11)

    dist1 = np.maximum(dist11, dist12)

    #print("dist1", dist1)

    # Frechest distance assuming 2 -> 1
    dist21 = np.linalg.norm(line[0:2] - lines2[:,0:2], axis=1)
    dist22 = np.linalg.norm(line[2:4] - lines2[:,2:4], axis=1)

    dist2 = np.maximum(dist21, dist22)

    # Smallest Frechet distance for each line
    dists = np.minimum(dist1, dist2)

    #print("dists", dists)

    return dists

def compute_line_distances(lines1, lines2):
    dists = np.zeros((len(lines1), len(lines2)))
    #print(lines1, lines2)

    for i in range(dists.shape[1]):
        line2 = lines2[i]
        dists[:, i] = compute_line_distance(line2, lines1)

    return dists


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: a np array of boxes
    For better performance, pass the largest set first and the smaller second.

    :return: a matrix of overlaps [boxes1 count, boxes2 count]
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.

    boxes1 = convert_format(boxes1)
    boxes2 = convert_format(boxes2)
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1)
    return overlaps


def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.  pred_boxes: a list of predicted Polygons of size N
    gt_boxes: a list of ground truth Polygons of size N
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons = convert_format(boxes)

    top = 64
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:64]
    
    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)

def decode_pred(pred, config, bev, dataset):
    if pred.size(0) != 1:
        raise ValueError("Wrong tensor size {}".format(pred.size()))

    pred = dataset.reg_target_denormalize(pred)
     
    pred = pred.squeeze().permute(1, 2, 0).numpy()
    class_map, reg_map  = pred[:,:,0], pred[:,:,1:]

    lines, scores = bev.decode_reg_map(reg_map, class_map, config["cls_threshold"])

def compute_similarity(line, lines):
    # Similarity is 0 when directions are opposite and 1 when they are the same
    dir_sim = 1 - (np.arccos(np.sum(line[0:2] * lines[:,0:2], axis=1)) / np.pi)
    max_dist = 0.3
    dist_sim = 1 - (np.clip(lines[:, 2] - line[2], 0, 0.3) / 0.3)

    return dir_sim + dist_sim


def line_non_max_suppression(lines, scores, threshold):
    if len(lines) == 0:
        return lines

    top = 64

    # Get indicies of boxes sorted by scores (highest first)
    idxs = scores.argsort()[::-1][:64]

    pick = []
    while len(idxs) > 0:
        # Pick top box and add its index to the list
        i = idxs[0]
        pick.append(i)

        # Compute similarity of the picked line with the rest
        sim = compute_similarity(lines[i], lines[idxs[1:]])

        # Identify boxes with similarity over the threshold. This
        # returns indices into idxs[1:], so add 1 to get
        # indices into idxs.

        remove_idxs = np.where(sim > threshold)[0] + 1

        # Remove indices of the picked and overlapped boxes.
        idxs = np.delete(idxs, remove_idxs)
        idxs = np.delete(idxs, 0)

    return np.array(pick, dtype=np.int32)


def filter_pred(config, pred):
    if len(pred.size()) == 4:
        if pred.size(0) == 1:
            pred.squeeze_(0)
        else:
            raise ValueError("Tensor dimension is not right")

    cls_pred = pred[0, ...]
    activation = cls_pred > config['cls_threshold']
    num_boxes = int(activation.sum())

    if num_boxes == 0:
        #print("No bounding box found")
        return [], []

    corners = torch.zeros((num_boxes, 8))
    for i in range(7, 15):
        corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
    corners = corners.view(-1, 4, 2).numpy()
    scores = torch.masked_select(cls_pred, activation).cpu().numpy()

    # NMS
    selected_ids = non_max_suppression(corners, scores, config['nms_iou_threshold'])
    corners = corners[selected_ids]
    scores = scores[selected_ids]

    return corners, scores


def compute_ap_range(gt_box, gt_class_id,
                     pred_box, pred_class_id, pred_score,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id,
                       pred_box, pred_class_id, pred_score,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP

def compute_ap(pred_match, num_gt, num_pred):

    assert num_gt != 0

    if num_pred == 0:
        return np.nan, np.nan, 0, np.nan, 0

    tp = (pred_match > -1).sum()
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(num_pred) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / num_gt

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    precision = tp / num_pred
    recall = tp / num_gt
    return mAP, precisions, recalls, precision, recall

def compute_precision_recall(
        pred_matches, num_gt, num_pred, distance_threshold=20):

    true_positives = np.count_nonzero(pred_matches != -1)
    false_positives = num_pred - true_positives
    false_negatives = num_gt - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall, true_positives, false_positives, false_negatives

def line_length(line):
    return np.linalg.norm(line[0:2] - line[2:4])

def compute_geometric_errors(gt_lines, pred_lines, gt_matches):
    tot = 0
    length_error = 0
    angle_error = 0
    for i in range(len(gt_lines)):
        match_idx = gt_matches[i]

        if match_idx != 0:
            pred_match = pred_lines[match_idx]
            gt_line = gt_lines[i]

            gt_len = line_length(gt_line)
            pred_len = line_length(pred_match)

            tot += 1

def compute_line_matches(gt_lines,
                    pred_lines,
                    distance_threshold=20):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted line.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        distances: [pred_lines, gt_lines] distances.
    """

    if len(pred_lines) == 0:
        return -1 * np.ones([gt_lines.shape[0]]), np.array([]), np.array([])

    # Compute IoU overlaps [pred_lines, gt_lines]
    distances = compute_line_distances(pred_lines, gt_lines)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_lines.shape[0]], dtype='int')
    gt_match = -1 * np.ones([gt_lines.shape[0]], dtype='int')
    for i in range(len(pred_lines)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(distances[i])

        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue

            # If we reach IoU smaller than the threshold, end the loop
            distance = distances[i, j]
            if distance > distance_threshold:
                break
            else:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, distances

def compute_matches(gt_boxes,
                    pred_boxes, pred_scores,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """

    if len(pred_scores) == 0:
        return -1 * np.ones([gt_boxes.shape[0]]), np.array([]), np.array([])

    gt_class_ids = np.ones(len(gt_boxes), dtype=int)
    pred_class_ids = np.ones(len(pred_scores), dtype=int)

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]

        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps
