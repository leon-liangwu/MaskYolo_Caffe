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

def draw_boxes(img, boxes, class_names):
    img_h, img_w, img_c = img.shape
    map_box = []

    for box in boxes:

        left = int((box[0] - box[2]/2) * img_w)
        top = int((box[1] - box[3]/2) * img_h)
        right = int((box[0] + box[2]/2) * img_w)
        bottom = int((box[1] + box[3]/2) * img_h)


        cv2.rectangle(img, (left, top), (right, bottom), (255,0,0),2) 
        map_box.append([box[4], left, top, right, bottom])

        # draw bar 
        right = max(right, left + 88)
        bar_size = 16
        points = ((left, top), (right, top), (right, top-bar_size), (left, top-bar_size))
        points = np.array(points)
        cv2.fillPoly(img, [points], (0, 0, 255))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cls_id = int(box[5])
        cls_name = class_names[cls_id]
        prob = box[4]
        text = '%s: %.3f' % (cls_name, prob)
        cv2.putText(img, text, (left, top-4), font, 0.38, (255, 255, 255))

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

    reg_list = np.zeros((pred_items.shape[0], 6))
    reg_list[..., 0] = (annc_items[..., 0] + pred_items[..., 0]) / out_w
    reg_list[..., 1] = (annc_items[..., 1] + pred_items[..., 1]) / out_h
    reg_list[..., 2] = (annc_items[..., 2] * pred_items[..., 2]) / out_w
    reg_list[..., 3] = (annc_items[..., 3] * pred_items[..., 3]) / out_h
    reg_list[..., 4] = pred_items[..., 4]
    reg_list[..., 5] = np.argmax(pred_items[..., 5:], axis=-1)

    return reg_list

@numba.jit(nopython=True)
def nms_kernal(keep, sort_ind, riou, nms_thresh):
    num = len(sort_ind)
    for i in range(num):
        ind_i = sort_ind[i]
        if keep[ind_i] == False:
            continue
        for j in range(i+1, num):
            ind_j = sort_ind[j]
            if keep[ind_j] == False:
                continue

            iou = riou[ind_i, ind_j]
            if iou > nms_thresh:
                keep[ind_j] = False

def nms_gpu_rotated(boxes, nms_thresh):
    prob_np = boxes[:, 4]
    sort_ind = prob_np.argsort()[::-1]
    boxes1 = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    boxes1[:, :4] = boxes[:, :4]
    boxes1 *= 1000
    boxes2 = boxes1.copy()

    iou = rbbx_overlaps(boxes1, boxes2)
    keep = np.ones(len(sort_ind), dtype=np.int32)
    keep = (keep == 1)
    nms_kernal(keep, sort_ind, iou, nms_thresh)

    return boxes[keep]




# mask utils
color_set = []
color_set.append([0,0,205])
color_set.append([34,139,34])
color_set.append([192,192,128])
color_set.append([165,42,42])
color_set.append([128,64,128])
color_set.append([204,102,0])
color_set.append([184,134,11])
color_set.append([0,153,153])
color_set.append([0,134,141])
color_set.append([184,0,141])
color_set.append([184,134,0])
color_set.append([184,134,223])


def draw_rois(img, rois, roi_labels, net_w, net_h):
    (h, w, c) = img.shape
    (num, _) = rois.shape
    for i in range(num):
        if roi_labels[i] == 1:
            bbox = rois[i, 1:]
            #print bbox
            (left, top, right, bottom) = (int(bbox[0] / net_w * w), int(bbox[1] / net_h * h), int(bbox[2] / net_w * w), int(bbox[3] / net_h * h))
            #  print (left, top, right, bottom)
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

def draw_mask(mask_prob, roi_labels):
    (num, cls, w, h) = mask_prob.shape
    for i in range(num):
        if roi_labels[i] == 1:
            mask_img = np.zeros((h, w, 1), np.uint8)
            mask_bool = mask_prob[i,0, :] < mask_prob[i, 1, :]
            mask_uint = mask_bool.astype(np.uint8)
            mask_img = mask_uint * 255

            cv2.imshow('mask', mask_img)
            cv2.waitKey(0)

def resize_mask(mask, size):
    (dst_h, dst_w) = size
    (m_h, m_w, m_c) = mask.shape
    mk_resize = np.zeros((dst_h, dst_w, m_c), np.uint8)
    (img_h, img_w, img_c) = mask.shape
    for r in range(dst_h):
        for c in range(dst_w):
            (o_r, o_c) = (int(1.0 * img_h / dst_h * r), int(1.0 * img_w / dst_w * c))
            for i in range(m_c):
                mk_resize[r, c, i] = mask[o_r, o_c, i]

    return mk_resize

def draw_mask_on_roi(img, bbox, mask_img, mask_uint):
    (m_h, m_w, m_c) = mask_img.shape
    (left, top, right, bottom) = (bbox[0], bbox[1], bbox[2], bbox[3])
    bbox_w = right - left
    bbox_h = bottom - top
    mask_resized = mask_img
    alpha = 0.3

    mask_rev = (np.ones(mask_resized.shape, np.uint8) - np.stack((mask_uint, mask_uint, mask_uint), axis=2)).astype(np.uint8)
    img[top: bottom, left:right, :] = img[top: bottom, left:right, :]* alpha + (img[top: bottom, left:right, :] * mask_rev + mask_resized) * ( 1- alpha)


def drraw_rois_masks(img, rois, mask_prob, roi_labels, net_w, net_h):
    (h, w, c) = img.shape
    (num, _) = rois.shape
    (num, cls, m_w, m_h) = mask_prob.shape
    for i in range(num):
        if roi_labels[i] == 1:
            bbox = rois[i, 1:]
            (left, top, right, bottom) = (int(bbox[0] / net_w * w), int(bbox[1] / net_h * h), int(bbox[2] / net_w * w), int(bbox[3] / net_h * h))

            mask_img = np.zeros(( bottom - top, right - left, 3), np.uint8)
            mask_bool = (mask_prob[i, 1, :] * 225).astype(np.uint8)
            mask_bool = cv2.resize(mask_bool, (right - left, bottom - top))
            mask_bool = mask_bool[:] > (255 * 0.5)
            mask_uint = mask_bool.astype(np.uint8)


            mask_img[:, :, 0] = mask_uint * color_set[i%12][0]
            mask_img[:, :, 1] = mask_uint * color_set[i%12][1]
            mask_img[:, :, 2] = mask_uint * color_set[i%12][2]
            draw_mask_on_roi(img, (left, top, right, bottom), mask_img, mask_uint)