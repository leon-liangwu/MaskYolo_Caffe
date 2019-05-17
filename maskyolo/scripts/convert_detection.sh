#!/usr/bin/env sh

CAFFE_ROOT=./

RESIZE_W=448
RESIZE_H=448

# 2007 + 2012 trainval
LIST_FILE="/path/to/voc_train.txt"
LMDB_DIR="/path/to/lmdb/voc_train_lmdb"
SHUFFLE=true

$CAFFE_ROOT/build/tools/convert_dec_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
	$LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

