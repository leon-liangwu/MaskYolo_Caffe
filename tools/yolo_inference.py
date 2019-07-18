#coding=utf-8
import _init_paths
import datetime
import time
import cv2
import numpy as np
import sys
import math
import fire

from nms.gpu_nms import gpu_nms

sys.path.insert(0, '../caffe-maskyolo/python')

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)


classes = ["person", "car", "bus", "bicycle", "motorbike"]

nms_thresh = 0.4
box_num = 5
use_fpn = 1
box_thresh = 0.4
cls_num = len(classes)

biases = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

(input_w, input_h) = input_size = (304, 224)

#biases = [4.33, 8.16, 7.99, 20.72, 17.63, 34.76, 24.84, 72.31, 51.82, 117.13, 122.16, 172.49]
#biases = [5.03, 8.16, 11.41, 20.72, 25.19, 34.76, 35.49, 72.31, 74.03, 117.13, 174.51, 172.49]
biases = [8.00, 10.27, 15.51, 26.77, 35.73, 44.89, 59.50, 103.39, 160.69, 162.92]

box_num = len(biases) / 2
deploy = "../models/mb-v2-t4-cls5-yolo/mb-v2-t4-cls5.prototxt" 
caffe_model = "../models/mb-v2-t4-cls5-yolo/mb-v2-t4-cls5.caffemodel"


def logst_a(x):
  return 1./(1. + np.exp(-x))

def softmax(x):
  x = np.exp(x)
  sum_x = np.sum(x)
  x = x / sum_x
  return x

def get_region_box(x, anchors, i, j, w, h):
  x[0] = ((i + logst_a(x[0])) / w)
  x[1] = ((j + logst_a(x[1])) / h)
  if use_fpn:
    x[2] = (math.exp(x[2]) * anchors[0] / input_w) 
    x[3] = (math.exp(x[3]) * anchors[1] / input_h) 
  else:
    x[2] = (math.exp(x[2]) * anchors[0] / w) 
    x[3] = (math.exp(x[3]) * anchors[1] / h) 

def cal_iou(box1, box2):
  left = max(box1[0], box2[0])
  top = max(box1[1], box2[1])
  right = min(box1[2], box2[2])
  bottom = min(box1[3], box2[3])

  if left > right or top > bottom:
    return 0
  else:
    u =  (right - left) * (bottom - top)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return 1.0 * u / (a1 + a2 - u)

def do_nms(boxes, nms_thresh):
  #boxes = boxes.sort()
  boxes = boxes[np.lexsort(-boxes.T)]
  (num, coord) = boxes.shape
  
  count = 0
  for i in range(0, num):
    if boxes[i, 4] == 0:
      continue
    for j in range(i+1, num):
      if boxes[j, 4] == 0:
        continue
      if cal_iou(boxes[i], boxes[j]) > nms_thresh:
        boxes[j, 4] = 0
  boxes = boxes[np.lexsort(-boxes.T)]
  if boxes[num-1, 4] != 0:
    return boxes
  for i in range(0, num):
    if boxes[i, 4] != 0:
      continue
    else:
      return boxes[0:i+1, :]

def get_boxes(pred_out, img_org, box_thresh, test_cls):
  (h, w, c) = pred_out.shape
  pred_boxes = pred_out.copy()

  for j in range(h):
    for i in range(w):
      for n in range(box_num):
        pred_boxes[j, i, n*(5+cls_num)+5: n*(5+cls_num)+5+cls_num] = softmax(pred_boxes[j, i, n*(5+cls_num)+5: n*(5+cls_num)+5+cls_num])
        cls_vec = pred_boxes[j, i, n*(5+cls_num)+5: n*(5+cls_num)+5+cls_num]

        #print pred_boxes[j, i, n*(5+cls_num)+5: n*(5+cls_num)+5+cls_num]
        max_prob = np.max(cls_vec)
        #print cls_vec
        #print max_prob
        #pred_boxes[j, i, (5+cls_num)*n+4] = logst_a(pred_boxes[j, i, (5+cls_num)*n+4])# * max_prob
        scale = logst_a(pred_boxes[j, i, n*(5+cls_num)+4])
        pred_boxes[j, i, n*(5+cls_num)+4] = scale * max_prob

        #pred_boxes[j, i, n*(5+cls_num)+4] = scale

        cls_index = np.where(cls_vec == max_prob)
        #print pred_boxes[j, i, n*(5+cls_num)+4]
        
        if cls_index[0][0] == test_cls and pred_boxes[j, i, n*(5+cls_num)+4] >= box_thresh:
        #if pred_boxes[j, i, n*(5+cls_num)+4] >= box_thresh:
          #print scale
          get_region_box(pred_boxes[j, i, n*(5+cls_num):n*(5+cls_num)+4], biases[2*n:2*(n+1)], i, j, w, h)
          #print pred_boxes[j, i, n*(5+cls_num):n*(5+cls_num)+4]
        else:
          pred_boxes[j, i, n*(5+cls_num)+4] = 0

   
  pred_boxes = pred_boxes.reshape((h*w*box_num, 5+cls_num))
  (img_h, img_w, imgc) = img_org.shape
  for i in range(0, h*w*box_num):
    if pred_boxes[i, 4] >= box_thresh:
      #max_prob = pred_boxes[i, 5+test_cls]
      #pred_boxes[i, 4] = max_prob
      box = pred_boxes[i]
      left = (box[0] - box[2]/2) * img_w
      top = (box[1] - box[3]/2) * img_h
      right = (box[0] + box[2]/2) * img_w
      bottom = (box[1] + box[3]/2) * img_h

      left = int(left if left > 0 else 0)
      top = int(top if top > 0 else 0)
      right = int(right if right < img_w else (img_w - 1))
      bottom = int(bottom if bottom < img_h else (img_h - 1))
      pred_boxes[i, 0:4] = [left, top, right, bottom]
    else:
      pred_boxes[i, 4] = 0
  return pred_boxes[:, 0:5]


def get_box_and_nms(pred_val, img_org, box_thresh, test_cls):
  boxes = get_boxes(pred_val, img_org, box_thresh, test_cls)
  #boxes = do_nms(boxes, nms_thresh)
  keep = gpu_nms(boxes, nms_thresh, device_id=0)
  if keep == None:
    return []
  boxes = boxes[keep, :]
  return boxes

def convert_boxes(img_org, boxes):
  map_box=[]
  num = boxes.shape[0]
  (img_h, img_w, imgc) = img_org.shape
  for i in range(0, num):
    box = boxes[i]
    if box[4] < box_thresh:
      continue

    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]


    cv2.rectangle(img_org, (left, top), (right, bottom), (255,0,0),2) 
    map_box.append([box[4], left, top, right, bottom])

  return map_box


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

  print pred_val.shape

  for test_cls in range(len(classes)):
    boxes = get_box_and_nms(pred_val, img_org, box_thresh, test_cls)
    boxes = convert_boxes(img_org, boxes)

  cv2.imshow('Oto Video', img_org) #显示
  cv2.waitKey(0) #延迟

if __name__ == '__main__':
  fire.Fire(run)


