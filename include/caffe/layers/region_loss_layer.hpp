#ifndef CAFFE_REGION_LOSS_LAYER_HPP_
#define CAFFE_REGION_LOSS_LAYER_HPP_

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
class RegionLossLayer : public LossLayer<Dtype> {
 public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionLoss"; }

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
  int coords_;
  int classes_;

  float object_scale_;
  float noobject_scale_;
  float class_scale_;
  float coord_scale_;

  bool softmax_;
  bool rescore_;

  int batch_;
  int outputs_;
  int inputs_;
  int truths_;
  int lab_count_;
  int max_box_num_;

  float thresh_;
  bool bias_match_;
  bool with_mask_;
  bool with_rcnn_;
  int mask_w_;
  int mask_h_;
  int label_stride_;

  Blob<Dtype> biases_;
  Blob<Dtype> output_;
  Blob<Dtype> diff_;
  int num_train_;
  int display_inter_;
};

}  // namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_
