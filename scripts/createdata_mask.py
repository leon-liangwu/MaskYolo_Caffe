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
from tqdm import tqdm




#nms = set([cat['supercategory'] for cat in cats])
#print('COCO supercategories: \n{}'.format(' '.join(nms)))


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
    




def run(coco_dir, lmdb_dir):
    (dst_h, dst_w) = [240, 320]
    lmdb_path = lmdb_dir
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)
    writeCount = 0

    dataTypes=['train', 'val']
    years=['2014', '2017']

    dataTypes=['val']
    years=[ '2017']

    dataDir = coco_dir


    for dataType in dataTypes:
        for year in years:

            imageSet = dataType + year

            ann_path = '{}/annotations/instances_{}.json'.format(dataDir, imageSet)
            coco = COCO(ann_path)

            # display COCO categories and supercategories
            cats = coco.loadCats(coco.getCatIds())
            nms=[cat['name'] for cat in cats]
            # print('COCO categories: \n{}\n'.format(' '.join(nms)))

            # get all images containing given categories, select one at random
            catIds = coco.getCatIds(catNms=['person']);
            imgIds = coco.getImgIds();
            #imgIds = coco.getImgIds(imgIds = [324158])
            # print type(imgIds)

            random.shuffle(imgIds)

            count_id = 0

            with tqdm(total=len(imgIds)) as pbar:
                pbar.set_description("Writing LMDB")
                for imgId in imgIds:
                    #if count_id >= 32:
                    #  break
                    count_id = count_id+1
                    img = coco.loadImgs(imgId)[0]

                    img_path = '%s/data/%s/%s' %(dataDir, imageSet, img['file_name'])
                    # print img_path

                    # load and display instance annotations
                    #plt.imshow(I); plt.axis('off')
                    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                    anns = coco.loadAnns(annIds)
                    #coco.showAnns(anns)
                    #plt.show()

                    im = cv2.imread(img_path)

                    (img_h, img_w, channel) = im.shape
                    mask = np.zeros((img_h, img_w, 1), np.uint8)


                    count = 0
                    boxes = []
                    M = 1
                    for ann in anns:
                        polys = []
                        if type(ann['segmentation']) == list:
                            count += 1
                            # polygon
                            for seg in ann['segmentation']:
                                poly = np.array(seg).reshape((int(len(seg) / 2)), 2).astype(np.int32)
                                polys.append(poly)
                                cv2.fillPoly(mask, [poly], (M * count, M * count, M * count))
                            boxes.append(ann['bbox'])

                        else:
                            # mask
                            t = img
                            if type(ann['segmentation']['counts']) == list:
                                rle = COCOmask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                            else:
                                rle = [ann['segmentation']]

                            m = COCOmask.decode(rle)
                            if ann['iscrowd'] == 1:
                                #m = m* (-1 )
                                m = m * 255
                            else:
                                count += 1
                                boxes.append(ann['bbox'])
                                m = m * count * M
                            mask = mask + m
                            #cv2.imshow('m',m)
                            #cv2.waitKey(0)

                    ct_boxes = []
                    for box in boxes:
                        (x, y, w, h) = box
                        box = [1.0*(x+w/2) / img_w, 1.0*(y+h/2)/img_h, 1.0*w/img_w, 1.0*h/img_h]
                        ct_boxes.append(box)



                    im_resize = cv2.resize(im, (dst_w, dst_h))
                    mk_resize = resize_mask(mask, (dst_h, dst_w))
                    # for box in ct_boxes:
                    #    drawbox(mk_resize, box)

                    img4ch = np.concatenate((im_resize, mk_resize), axis=2)
                    boxlab = convert_boxes_labels(ct_boxes, 1.0 * img_w / img_h)
                    #print boxlab

                    #cv2.imshow('img', im_resize)
                    #cv2.imshow('mask', mk_resize)
                    #cv2.waitKey(0)

                    #print img4ch.shape
                    img4ch = np.transpose(img4ch, (2, 0, 1))
                    #print img4ch.shape

                    datum = caffe.io.array_to_datum2(img4ch, boxlab)
                    key = '%07d' % writeCount
                    txn.put(key, datum.SerializeToString())
                    if (writeCount % 1000 == 0):
                        txn.commit()
                        txn = env.begin(write=True)
                        # print 'write count: %d' % (writeCount)
                    writeCount = writeCount + 1
                    pbar.update(1)

    txn.commit()
    env.close()

if __name__ == '__main__':
    fire.Fire(run)
