#!/usr/bin/env sh

CAFFE_HOME=../../caffe-maskyolo

SOLVER=./solver_step2.prototxt
WEIGHTS="/data/Machine_Learning/models/mb_body_mask_step1_iter_40000.caffemodel"

$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER  --weights=$WEIGHTS --gpu=0  #--gpu=0,1,2,3 for multigpus

