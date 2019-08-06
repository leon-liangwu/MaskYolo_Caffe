#!/usr/bin/env sh

CAFFE_ROOT=./caffe-maskyolo

RESIZE_W=448
RESIZE_H=448

# 2007 + 2012 trainval
LIST_FILE=$1
LMDB_DIR=$2
SHUFFLE=true

$CAFFE_ROOT/build/tools/convert_dec_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
	$LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

