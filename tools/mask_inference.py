#coding=utf-8
import os
import numpy as np
import cv2
import time
import fire

import _init_paths
from utils import draw_rois, drraw_rois_masks

box_thresh = 0.60
nms_thresh = 0.4
cls_num = 0
box_num = 3
IMAGE_SIZE = (320, 224)

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

def_img_path = '../assets/000000401862.jpg'
def_model = '../models_maskyolo/mb_body_mask/mb_body_mask_deploy.prototxt'
def_weights = '../models_maskyolo/pretrained_models/mb_body_mask.caffemodel'

def run(img_path=def_img_path, model=def_model, weights=def_weights):
    net = caffe.Net(model, weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))

    img = cv2.imread(img_path)

    (height, width, channel) = img.shape
    img_org = img

    img = img.astype(np.float32)

    scale = 1. / 128
    img = img[:] - 128
    img = img[:] * scale
    img = cv2.resize(img, IMAGE_SIZE)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    (net_b, net_c, net_h, net_w)  = net.blobs['data'].data.shape

    t1 = time.time()
    out = net.forward()
    t2 = time.time()
    total_time = t2 - t1
    print("Net forward time consumed: %.2fms" % (total_time * 100))

    rois = net.blobs['rois'].data
    roi_labels = net.blobs['roi_labels'].data

    img_res = img_org.copy()

    draw_rois(img_res, rois, roi_labels, net_w, net_h)
    mask_prob = out['mask_prob']

    drraw_rois_masks(img_res, rois, mask_prob, roi_labels, net_w, net_h)

    img_dir, img_name = os.path.split(img_path)
    save_name = img_name.replace('.', '_mask.')
    cv2.imwrite('results/'+save_name, img_res)
    save_path = 'results/'+save_name
    cv2.imwrite('results/'+save_name, img_res)
    print("Result saved in %s!" % save_path)

    cv2.imshow('mask demo', img_res)
    cv2.waitKey(0)



if __name__ == '__main__':
    fire.Fire(run)
