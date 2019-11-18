#!/usr/bin/env sh

CAFFE_HOME=../../caffe-maskyolo

SOLVER=./solver_step1.prototxt
WEIGHTS="../pretrained_models/mb_v2_t4_yolo.caffemodel"

$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER  --weights=$WEIGHTS --gpu=0  #--gpu=0,1,2,3 for multigpus

