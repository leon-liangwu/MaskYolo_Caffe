# YOLO Caffe version with MaskRCNN

### Caffe-MaskYolo

#### What I add in this version of caffe?
- [x] detection lmdb, mask lmdb, keypoint lmdb prepare
- [x] yolo v2 (RegionLossLayer) and v3 (YoloLossLayer) are supported
- [x] Instance Mask segmentation with Yolo
- [x] Pose Recognition with yolo

#### preparation
```
# install requirements
cd ROOT_MaskYolo
pip install -r requirements.txt
# compare caffe
cd caffe-maskyolo
cp Makefile.config.example Makefile.config
make -j
make pycaffe
```

### Object Detection with YOLO
support to use yolo v2 or v3 to detect objects in images
#### examples of detection on KITTI
![](assets/detection1.png)

#### prepare lmdb for object detection 
```
cd ROOT
sh ./maskyolo/scripts/convert_detection.sh  #generate lmdb for detection
cd ./maskyolo/models/mobilenetv2-yolo/
nohup sh yolo_train.sh > train.log &
tail -f train.log
```

#### objection demo
```
cd maskyolo
python demos/yolo_inference.py --model=./models/ --weights=./models
```


### Instance mask and Keypoint recognization with MaskRCNN combined YOLO

Use yolo results as input to feed to `roi_pooing` or `roi_alignment` layer 
#### examples of mask & keypoints on COCO
![](assets/mask_keypoints.png)

#### prepare lmdb for mask regression
```
coming soon
```

#### yolo with mask demo
```
comming soon
```

### Reference

> You Only Look Once: Unified, Real-Time Object detection http://arxiv.org/abs/1506.02640

> YOLO9000: Better, Faster, Stronger https://arxiv.org/abs/1612.08242

> YOLOv3: An Incremental Improvement https://pjreddie.com/media/files/papers/YOLOv3.pdf

> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

> Mask R-CNN 

