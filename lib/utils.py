import cv2
import numpy as np
import numba
import math
from box_utils.rbbox_overlaps import rbbx_overlaps

def generate_anchors(output_size, box_num, biases):
    out_w, out_h = output_size
    anchors = np.zeros((out_h, out_w, box_num, 4), dtype=np.float32)

    x_range = np.arange(out_w)
    y_range = np.arange(out_h)
    x_map, y_map = np.meshgrid(x_range, y_range)
    x_map = np.expand_dims(x_map, axis=2)
    y_map = np.expand_dims(y_map, axis=2)

    anchors[..., 0] = x_map
    anchors[..., 1] = y_map

    for i in range(box_num):
        anchors[..., i, 2:] = biases[i*2 +0: i*2 + 2]

    return anchors

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def softmax(x):
    x = np.exp(x)
    sum_x = np.sum(x, axis=-1, keepdims=True)
    x = x / sum_x
    return x

def draw_boxes(img, boxes):
    img_h, img_w, img_c = img.shape
    map_box = []

    for box in boxes:

        left = int((box[0] - box[2]/2) * img_w)
        top = int((box[1] - box[3]/2) * img_h)
        right = int((box[0] + box[2]/2) * img_w)
        bottom = int((box[1] + box[3]/2) * img_h)


        cv2.rectangle(img, (left, top), (right, bottom), (255,0,0),2) 
        map_box.append([box[4], left, top, right, bottom])
    return map_box

def decode_boxes(pred_val, anchors, cls_num, nms_thresh, box_thresh):
    out_h, out_w, out_c = pred_val.shape
    out_h, out_w, box_num, ann_c = anchors.shape
    reg_c = out_c / box_num
    if cls_num > 0:
        pred_val = pred_val.reshape((out_h, out_w, box_num, reg_c))

    pred_val[..., 4] = sigmoid(pred_val[..., 4])

    pred_map = pred_val[..., 4]
    
    items = np.where(pred_map > box_thresh)

    pred_items = pred_val[items]
    annc_items = anchors[items]
    pred_items[..., :2] = sigmoid(pred_items[..., :2])
    pred_items[..., 2:4] = np.exp(pred_items[..., 2:4])

    pred_items[..., 5:] = softmax(pred_items[..., 5:])
    max_prob = np.max(pred_items[..., 5:])

    reg_list = np.zeros((pred_items.shape[0], 7))
    reg_list[..., 0] = (annc_items[..., 0] + pred_items[..., 0]) / out_w
    reg_list[..., 1] = (annc_items[..., 1] + pred_items[..., 1]) / out_h
    reg_list[..., 2] = (annc_items[..., 2] * pred_items[..., 2]) / out_w
    reg_list[..., 3] = (annc_items[..., 3] * pred_items[..., 3]) / out_h
    reg_list[..., 4] = pred_items[..., 4]
    reg_list[..., 5] = np.max(pred_items[..., 5:])

    return reg_list

@numba.jit(nopython=True)
def nms_kernal(keep, sort_ind, riou, nms_thresh):
    num = len(sort_ind)
    for i in range(num):
        ind_i = sort_ind[i]
        if keep[ind_i] == 0:
            continue
        for j in range(i+1, num):
            ind_j = sort_ind[j]
            if keep[ind_j] == 0:
                continue

            iou = riou[ind_i, ind_j]
            if iou > nms_thresh:
                keep[ind_j] = 0

def nms_gpu_rotated(boxes, nms_thresh):
    prob_np = boxes[:, 4]
    sort_ind = prob_np.argsort()[::-1]
    boxes1 = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    boxes1[:, :4] = boxes[:, :4]
    boxes2 = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    boxes2[:, :4] = boxes[:, :4]

    iou = rbbx_overlaps(boxes1, boxes2)
    keep = np.ones(len(sort_ind), dtype=np.int32)
    nms_kernal(keep, sort_ind, iou, nms_thresh)

    return boxes[keep]



