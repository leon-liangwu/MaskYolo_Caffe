import sys

sys.path.insert(0, './caffe-maskyolo/python/')
sys.path.insert(0, './lib/cocoapi/PythonAPI/')

import caffe
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import random

import lmdb
import fire


def drawbox(img, box):
    (img_h, img_w, img_c) = img.shape
    (x, y, w, h) = box
    p0 = (int((x - w/2) * img_w), int((y - h/2) * img_h))
    p1 = (int((x + w/2) * img_w), int((y + h/2) * img_h))
    cv2.rectangle(img, p0, p1, (255,255,255), 1)

def convert_boxes_labels(ct_boxes, ratio):
    boxlab = np.zeros((len(ct_boxes) * 6 + 1), np.float32)
    for i in range(len(ct_boxes)):
        box = ct_boxes[i]
        boxlab[i * 6 + 0] = 0
        boxlab[i * 6 + 1: i * 6 + 5] = box
        boxlab[i * 6 + 5] = i + 1
    boxlab[len(ct_boxes) * 6 + 0] = ratio
    return boxlab

def resize_mask(mask, size):
    (dst_h, dst_w) = size
    mk_resize = np.zeros((dst_h, dst_w, 1), np.uint8)
    (img_h, img_w, img_c) = mask.shape
    for r in range(dst_h):
        for c in range(dst_w):
            (o_r, o_c) = (int(1.0 * img_h / dst_h * r), int(1.0 * img_w / dst_w * c))
            mk_resize[r, c, 0] = mask[o_r, o_c, 0]

    return mk_resize

def asign_kps(kp_resize, ki_resize, keypoints, org_size):
    (h, w, _) = kp_resize.shape
    (oh, ow, _) = org_size
    (r_h, r_w) = (1.0 * h / oh, 1.0 * w / ow)
    for k in keypoints.keys():
        kp = keypoints[k]
        for i in range(kp.shape[0]/3):
            (ox, oy, v) = (kp[i*3 + 0], kp[i*3 + 1], kp[i*3 + 2])
            if ox > 0 and oy > 0:
                (x, y) = (int(ox * r_w), int(oy * r_h))
                ki_resize[y, x] = k
                kp_resize[y, x] = i + 1

def decodeim6ch(img6ch, sks):
    im6ch = np.transpose(img6ch, (1, 2, 0))
    #print img6ch.shape
    #print im6ch.shape
    im = im6ch[:, :, :3].copy()
    mask = im6ch[:, :, 3]
    kp_ind = im6ch[:, :, 4]
    kp_cls = im6ch[:, :, 5]
    #cv2.imshow('img', im)
    #cv2.imshow('mask', mask)

    kps = {}
    (w, h, _) = im6ch.shape
    #print kp_ind.shape
    for r in range(h):
        for c in range(w):
            if kp_ind[r, c] > 0:
                ind = kp_ind[r, c]
                cls = kp_cls[r, c]
                #print (ind, cls)
                if not kps.has_key(ind):
                    kps[ind] = np.zeros((17, 2), dtype=np.uint32)
                kps[ind][cls - 1, 0] = r
                kps[ind][cls - 1, 1] = c
                #print (ind, c, r)

    #cv2.imshow('kp_ind', kp_ind * 50)
    for k in kps.keys():
        for i in range(kps[k].shape[0]):
            if kps[k][i, 0] > 0 and kps[k][i, 1] > 0:
                #print  (kps[k][i, 1], kps[k][i, 0])
                cv2.circle(im, (kps[k][i, 1], kps[k][i, 0]), 2, (0, 0, 255), 2)

    #cv2.imshow('img', im)
    #cv2.imshow('kp_cls', kp_cls * 50)
    #cv2.waitKey(0)


def run(coco_dir, lmdb_dir):
        
    (dst_h, dst_w) = [320, 320]
    lmdb_path = lmdb_dir
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)
    writeCount = 0

    #dataTypes=['train', 'val']
    #years=['2014', '2017']

    dataTypes=['train']
    years=[ '2017']

    dataDir = coco_dir

    mag = 1


    for dataType in dataTypes:
        for year in years:

            imageSet = dataType + year

            ann_path = '{}/annotations/instances_{}.json'.format(dataDir, imageSet)
            kps_path = '{}/annotations/person_keypoints_{}.json'.format(dataDir, imageSet)
            coco = COCO(ann_path)
            coco_kps = COCO(kps_path)

            # display COCO categories and supercategories
            cats = coco.loadCats(coco.getCatIds())
            nms=[cat['name'] for cat in cats]
            print('COCO categories: \n{}\n'.format(' '.join(nms)))

            # get all images containing given categories, select one at random
            catIds = coco.getCatIds(catNms=['person']);
            imgIds = coco.getImgIds();
            #imgIds = coco.getImgIds(imgIds = [324158])

            #random.shuffle(imgIds)

            count_id = 0

            for imgId in imgIds:
                #if count_id >= 32:
                #  break
                count_id = count_id+1
                img = coco.loadImgs(imgId)[0]

                img_path = '%s/data/%s/%s' %(dataDir, imageSet, img['file_name'])
                print(img_path)

                # load and display instance annotations
                #plt.imshow(I); plt.axis('off')
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)
                annIds_kps = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns_kps = coco_kps.loadAnns(annIds_kps)
                #coco.showAnns(anns)
                #plt.show()

                #coco_kps.showAnns(anns_kps)
                #plt.show()

                im = cv2.imread(img_path)

                (img_h, img_w, channel) = im.shape
                mask = np.zeros((img_h, img_w, 1), np.uint8)

                count = 0
                boxes = []
                keypoints = {}
                sks = 0

                for ann, ann_kps in zip(anns, anns_kps):
                    polys = []
                    if type(ann_kps['segmentation']) == list:
                        count += 1
                        # polygon
                        for seg in ann_kps['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2)), 2).astype(np.int32)
                            polys.append(poly)
                            cv2.fillPoly(mask, [poly], (mag * count, mag * count, mag * count))
                        boxes.append(ann_kps['bbox'])

                    else:
                        # mask
                        t = img
                        if type(ann_kps['segmentation']['counts']) == list:
                            rle = COCOmask.frPyObjects([ann_kps['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann_kps['segmentation']]

                        m = COCOmask.decode(rle)
                        if ann_kps['iscrowd'] == 1:
                            #m = m* (-1 )
                            m = m * 255
                        else:
                            count += 1
                            boxes.append(ann_kps['bbox'])
                            m = m * count * mag
                        mask = mask + m
                        #cv2.imshow('m',m)
                        #cv2.waitKey(0)

                    if count ==0:
                        continue
                    sks = np.array(coco_kps.loadCats(ann_kps['category_id'])[0]['skeleton']) - 1

                    kp = np.array(ann_kps['keypoints'])

                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]

                    keypoints[count] = kp

                    '''
                    kp = kp.reshape((17, 3))
                    for sk in sks:
                    if kp[sk[0], 0] != 0 and kp[sk[0], 1] != 0 \
                        and kp[sk[0], 0] != 0 and kp[sk[1], 1] != 0:
                        cv2.line(im, (kp[sk[0],0], kp[sk[0],1]), (kp[sk[1],0], kp[sk[1],1]), (0, 255, 255),
                                2)

                    cv2.imshow("iimg", im)
                    cv2.waitKey(0)

                    
                    for sk in sks:
                    if np.all(v[sk] > 0):
                        cv2.line(im, (kp[sk[0]*3 + 0], kp[sk[0]*3 + 1]), (kp[sk[1]*3 + 0], kp[sk[1]*3 + 1]), (0,255,255), 2)

                    for px, py in zip(x, y):
                    if px > 0 or py > 0:
                        cv2.circle(im, (px, py), 2, (0, 0, 255), 5)
                    '''
                    '''
                    c = [0.1, 0.2, 0.8]
                    for sk in sks:
                    if np.all(v[sk] > 0):
                        plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
                    plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
                    plt.show()
                    '''

                ct_boxes = []
                for box in boxes:
                    (x, y, w, h) = box
                    box = [1.0*(x+w/2) / img_w, 1.0*(y+h/2) / img_h, 1.0*w / img_w, 1.0*h / img_h]
                    ct_boxes.append(box)


                im_resize = cv2.resize(im, (dst_w, dst_h))
                mk_resize = resize_mask(mask, (dst_h, dst_w))
                #for box in ct_boxes:
                #  drawbox(mk_resize, box)


                kp_resize = np.zeros(mk_resize.shape, dtype=np.uint8)
                ki_resize = np.zeros(mk_resize.shape, dtype=np.uint8)
                asign_kps(kp_resize, ki_resize, keypoints, mask.shape)


                img6ch = np.concatenate((im_resize, mk_resize, ki_resize, kp_resize), axis=2)
                boxlab = convert_boxes_labels(ct_boxes, 1.0 * img_w / img_h)

                #cv2.imshow('img', im_resize)
                #cv2.imshow('mask', mk_resize)
                #cv2.imshow('keypoints', kp_resize*20)

                #@cv2.waitKey(0)


                img6ch = np.transpose(img6ch, (2, 0, 1))
                decodeim6ch(img6ch, sks)

                datum = caffe.io.array_to_datum2(img6ch, boxlab)
                key = '%07d' % writeCount
                txn.put(key, datum.SerializeToString())
                if (writeCount % 1000 == 0):
                    txn.commit()
                    txn = env.begin(write=True)
                print('write count: %d' % (writeCount))
                writeCount = writeCount + 1


    txn.commit()
    env.close()


if __name__ == '__main__':
    fire.Fire(run)