import numpy as np
import cv2

img_list = open('/data/ImageSets/hb_tl/train.txt')
net_w, net_h = 608, 608
num_cls = 3

lines = img_list.readlines()
img_list.close()

def draw_centers(img, centers):
    img_h, img_w, img_c = img.shape
    cent_x = int(img_w / 2)
    cent_y = int(img_h / 2)
    for box in centers:
        w = box[0] * img_w
        h = box[1] * img_h
        left = int(cent_x - w / 2)
        top = int(cent_y - h / 2)
        right = int(cent_x + w / 2)
        bottom = int(cent_y + h / 2)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)


def cal_iou(b1, b2):
    a1 = b1[0] * b1[1]
    a2 = b2[0] * b2[1]

    inter = min(b1[0], b2[0]) * min(b1[1], b2[1])
    return inter / (a1+a2-inter)

def update_class(box_array, cls_array, centers):
    box_num, _ = box_array.shape
    cls_num, _ = centers.shape

    for i in range(box_num):
        min_dist = 0
        for j in range(cls_num):
            dist = -cal_iou(box_array[i], centers[j])
            if dist < min_dist:
                min_dist = dist
                cls_array[i] = j

    for i in range(cls_num):
        centers[i] = np.mean(box_array[cls_array==i], axis=0)


def total_dist(box_array, cls_array, centers):
    box_num, _ = box_array.shape
    cls_num, _ = centers.shape
    dist = 0

    for i in range(box_num):
        dist += -cal_iou(box_array[i], centers[cls_array[i]])

    return dist

whs = []
for line in lines:
    img_path = line.replace('\n', '')
    lab_path = img_path.replace('images', 'labels')
    lab_path = lab_path.replace('jpg', 'txt')


    lab_txt = open(lab_path)
    lab_lines = lab_txt.readlines()

    for lab_line in lab_lines:
        entry = lab_line.split(' ')
        w, h = float(entry[3]), float(entry[4])
        whs.append([w, h])

box_array = np.array(whs)

cls_array = np.zeros(box_array.shape[0], dtype=np.int32)



centers = box_array[np.random.choice(box_array.shape[0], num_cls)]

update_class(box_array, cls_array, centers)
cur_dist = total_dist(box_array, cls_array, centers)

last_centers = np.zeros_like(centers)
iter = 0
value = np.abs(last_centers-centers).sum()
print 'iter : %d  cur_dist: %f value: %f' % (iter, cur_dist, value)
while value > 0.00001:

    last_centers = centers.copy()
    #print centers
    last_dist = cur_dist
    update_class(box_array, cls_array, centers)
    cur_dist = total_dist(box_array, cls_array, centers)
    value = np.abs(last_centers - centers).sum()
    print 'iter : %d  cur_dist: %f value: %f' % (iter, cur_dist, value)
    iter += 1


centers = centers[np.lexsort(centers[:, ::-1].T)]

img = np.zeros((net_w, net_h, 3), dtype=np.uint8)
draw_centers(img, centers)

final_centers = centers
final_centers[:, 0] = final_centers[:, 0] * net_w
final_centers[:, 1] = final_centers[:, 1] * net_h


for center in final_centers:
  print 'anchor_x: %3.2f' % (center[0])
  print 'anchor_y: %3.2f' % (center[1])

str_bias = 'bias ='
for center in final_centers:
    str_bias += ' ' + '%.2f' % (center[0]) + ',' + '%.2f' % (center[1])+','
print str_bias

cv2.imshow('img', img)
cv2.waitKey(0)





