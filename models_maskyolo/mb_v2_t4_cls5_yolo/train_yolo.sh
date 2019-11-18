#!/usr/bin/env sh

CAFFE_HOME=../../caffe-maskyolo/

SOLVER=solver.prototxt
WEIGHTS="../pretrained_models/imagenet_mb_v2_t4_iter_500000.caffemodel"

$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER  --weights=$WEIGHTS --gpu=0  #--gpu=0,1,2,3 for multigpus

