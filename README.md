# YOLO Caffe version with MaskRCNN

### Caffe-MaskYolo

#### What I add in this version of caffe?
- [x] detection lmdb, mask lmdb, keypoint lmdb prepare
- [x] yolo v2 (RegionLossLayer) and v3 (YoloLossLayer) are supported
- [x] Instance Mask segmentation with Yolo
- [x] Pose Recognition with yolo

#### preparation
```
# clone
git clone https://github.com/leon-liangwu/MaskYolo_Caffe.git --recursive

# install requirements
cd ROOT_MaskYolo
pip install -r requirements.txt

# compile box_utils
cd lib/box_utils
python setup.py build_ext --inplace

# compile caffe
cd caffe-maskyolo
cp Makefile.config.example Makefile.config
make -j
make pycaffe
```

#### download pretrained models
Click [here](https://www.dropbox.com/s/z1w2z8ya28v3lah/models.tgz?dl=0 "pretrained models") to download pretrained models
```
cd ROOT_MaskYolo
tar zxvf /your/downlaod/model/path/models.tgz ./
```


### Object Detection with YOLO
support to use yolo v2 or v3 to detect objects in images

#### objection demo
```
cd tools
python yolo_inference.py --model=../models/mb-v2-t4-cls5-yolo/mb-v2-t4-cls5.prototxt --weights=../models/mb-v2-t4-cls5-yolo/mb-v2-t4-cls5.caffemodel
# Net forward time consumed: 3.96ms
```
The demo result is shown below.

![](assets/detection1.png)

#### train for object detection 
```
cd ROOT_MaskYolo
sh ./scripts/convert_detection.sh  #generate lmdb for detection
cd ./models/mobilenetv2-yolo/
nohup sh yolo_train.sh > train.log &
tail -f train.log
```

### Instance Mask and Keypoints

#### yolo with mask demo
```
comming soon
```

Some resulting samples are show below. I use yolo results as input to feed to `roi_pooing` or `roi_alignment` layer.

![](assets/mask_keypoints.png)

#### prepare lmdb for mask regression
```
coming soon
```



### Reference

> You Only Look Once: Unified, Real-Time Object detection http://arxiv.org/abs/1506.02640

> YOLO9000: Better, Faster, Stronger https://arxiv.org/abs/1612.08242

> YOLOv3: An Incremental Improvement https://pjreddie.com/media/files/papers/YOLOv3.pdf

> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

> Mask R-CNN 

