#coding=utf-8
import _init_paths
import datetime
import time
import cv2
import numpy as np
import sys
import math
import fire

from box_utils.rbbox_overlaps import rbbx_overlaps
from utils import generate_anchors, decode_boxes, draw_boxes, nms_gpu_rotated

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

classes = ["person", "car", "bus", "bicycle", "motorbike"]

nms_thresh = 0.4
box_num = 5
use_fpn = 1
box_thresh = 0.4
cls_num = len(classes)

(input_w, input_h) = input_size = (304, 224)
(out_w, out_h) = output_size = (19, 14)
biases = [8.00, 10.27, 15.51, 26.77, 35.73, 44.89, 59.50, 103.39, 160.69, 162.92]
biases = np.array(biases, dtype=np.float32)
if use_fpn == 1:
    biases = 1.0 * biases  * out_w / input_w


box_num = len(biases) / 2
deploy = "../models/mb-v2-t4-cls5-yolo/mb-v2-t4-cls5.prototxt" 
caffe_model = "../models/mb-v2-t4-cls5-yolo/mb-v2-t4-cls5.caffemodel"

anchors = generate_anchors(output_size, box_num, biases)

def run(model, weights):
    net = caffe.Net(model,weights,caffe.TEST) 

    total_time = 0
    count = 0

    count = count + 1
    img = cv2.imread('../assets/000000.png')
    (height, width, channel) = img.shape

    img_org = img.copy()
    #img = cv2.resize(img, (480, 480))#, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
    (height, width, channel) = img.shape

    img = img.astype(np.float32)

    scale = 1. /128
    img = img[:] - 128
    img = img[:] * scale
    img = img.transpose((2,0,1))

    net.blobs['data'].data[...] = img 

    t1 = datetime.datetime.now()
    out = net.forward()
    t2 = datetime.datetime.now()
    total_time += 1.0 * (t2 - t1).microseconds / 1000

    pred_val = net.blobs['conv_out'].data[0]
    pred_val = pred_val.transpose((1,2,0))

    boxes = decode_boxes(pred_val.copy(), anchors, cls_num, nms_thresh, box_thresh)
    boxes = nms_gpu_rotated(boxes, nms_thresh)
    draw_boxes(img_org, boxes)

    # for test_cls in range(len(classes)):
    #     boxes = get_box_and_nms(pred_val, img_org, box_thresh, box_num, cls_num, biases, test_cls, nms_thresh)
    #     boxes = convert_boxes(img_org, boxes, box_thresh)

    cv2.imshow('Oto Video', img_org) #显示
    cv2.waitKey(0) #延迟

if __name__ == '__main__':
    fire.Fire(run)


