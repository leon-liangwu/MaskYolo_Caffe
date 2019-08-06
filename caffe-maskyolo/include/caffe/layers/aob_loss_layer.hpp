#ifndef CAFFE_AOB_LOSS_LAYER_HPP_
#define CAFFE_AOB_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);
template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth);
template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth);

template <typename Dtype>
class AobLossLayer : public LossLayer<Dtype> {
 public:
  explicit AobLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AobLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  int w_;
  int h_;
  int n_;
  int nw_;
  int nh_;
  int coords_;
  int classes_;
  int total_;

  int num_train_;

  float object_scale_;
  float noobject_scale_;
  float class_scale_;
  float coord_scale_;

  int batch_;
  int outputs_;
  int inputs_;
  int truths_;
  int max_box_num_;
  bool anchor_mask_;

  float ignore_thresh_;
  float ignore_w_;
  float ignore_h_;
  float truth_thresh_;

  int lab_count_;
  bool with_mask_;
  bool with_kps_;
  bool with_rcnn_;
  int mask_w_;
  int mask_h_;
  int label_stride_;

  vector<float> biases_;
  vector<int> mask_;

  Blob<Dtype> output_;
  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_AOB_LOSS_LAYER_HPP_
