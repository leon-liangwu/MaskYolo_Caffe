#ifndef CAFFE_DECODE_ROIS_LAYER_HPP_
#define CAFFE_DECODE_ROIS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class DecodeRoisLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with DecodeRoisLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit DecodeRoisLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- DecodeRoisLayer cannot be used as a loss.
   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int w_;
  int h_;
  int n_;
  int coords_;
  int classes_;
  int nw_;
  int nh_;

  int prop_num_;

  bool softmax_;

  int batch_;
  int outputs_;
  int inputs_;
  int truths_;
  int lab_count_;
  int target_size_;
  int max_box_num_;
  bool with_mask_;
  int mask_w_;
  int mask_h_;
  int label_stride_;
  int box_num_;

  float thresh_;
  float scale_;

  Blob<Dtype> biases_;
  Blob<Dtype> output_;
};

}  // namespace caffe

#endif  // CAFFE_DECODE_ROIS_LAYER_HPP_
