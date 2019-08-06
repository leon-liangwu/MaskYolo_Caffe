import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import fire

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = ["person", "car", "bus", "bicycle", "motorbike"]
#classes = []


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(voc_dir, year, image_id):
    in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(voc_dir, year, image_id))
    out_file = open('%s/VOCdevkit/VOC%s/labels/%s.txt'%(voc_dir, year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    flag_catch = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        flag_catch = 1
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return flag_catch


def run(voc_dir):

    for year, image_set in sets:
        if not os.path.exists('%s/VOCdevkit/VOC%s/labels/'%(voc_dir, year)):
            os.makedirs('%s/VOCdevkit/VOC%s/labels/'%(voc_dir, year))
        image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(voc_dir, year, image_set)).read().strip().split()
        list_file = open('%s/%s_%s.txt'%(voc_dir, year, image_set), 'w')
        for image_id in image_ids:
            if not convert_annotation(voc_dir, year, image_id):
                continue
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(voc_dir, year, image_id))
            convert_annotation(voc_dir, year, image_id)
        list_file.close()

if __name__ == '__main__':
    fire.Fire(run)

