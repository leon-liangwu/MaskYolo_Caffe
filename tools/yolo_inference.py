#coding=utf-8
import os
import time
import cv2
import numpy as np
import sys
import math
import fire

import _init_paths
from box_utils.rbbox_overlaps import rbbx_overlaps
from utils import generate_anchors, decode_boxes, draw_boxes, nms_gpu_rotated

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

classes = ["person", "car", "bus", "bicycle", "motorbike"]

nms_thresh = 0.4
box_num = 5
use_fpn = 1
box_thresh = 0.6
cls_num = len(classes)

(input_w, input_h) = input_size = (672, 224)
(out_w, out_h) = output_size = (input_w/16, input_h/16)
biases = [8.00, 10.27, 15.51, 26.77, 35.73, 44.89, 59.50, 103.39, 160.69, 162.92]
biases = np.array(biases, dtype=np.float32)
if use_fpn == 1:
    biases = 1.0 * biases  * out_w / input_w

box_num = len(biases) / 2

anchors = generate_anchors(output_size, box_num, biases)

def_img_path = '../assets/000000.png'
def_model = '../models_maskyolo/mb_v2_t4_cls5_yolo/mb_v2_t4_cls5_deploy.prototxt'
def_weights = '../models_maskyolo/pretrained_models/mb_v2_t4_cls5.caffemodel'

def run(img_path=def_img_path, model=def_model, weights=def_weights):
    net = caffe.Net(model, weights, caffe.TEST) 

    img = cv2.imread(img_path)
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

    t1 = time.time()
    out = net.forward()
    t2 = time.time()
    total_time = t2 - t1
    print("Net forward time consumed: %.2fms" % (total_time * 100))
    

    pred_val = net.blobs['conv_out'].data[0]
    pred_val = pred_val.transpose((1,2,0))

    boxes = decode_boxes(pred_val.copy(), anchors, cls_num, nms_thresh, box_thresh)
    boxes = nms_gpu_rotated(boxes, nms_thresh)

    img_res = img_org.copy()
    draw_boxes(img_res, boxes, classes)

    img_dir, img_name = os.path.split(img_path)
    save_name = img_name.replace('.', '_det.')
    save_path = 'results/'+save_name
    cv2.imwrite('results/'+save_name, img_res)
    print("Result saved in %s!" % save_path)

    cv2.imshow('yolo detection', img_res) #显示
    cv2.waitKey(0) #延迟


if __name__ == '__main__':
    fire.Fire(run)


