#!/usr/bin/env sh

CAFFE_HOME="../../../caffe-maskyolo/"

SOLVER=./det_solver.prototxt
WEIGHTS="../../pretrained_models/mb-reg-v2-T4-dw_iter_500000.caffemodel"

$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER  --weights=$WEIGHTS --gpu=0
