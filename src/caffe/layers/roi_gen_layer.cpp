#include <functional>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/layers/roi_gen_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/box.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include<cstdlib>
#include<ctime>

using namespace cv;
using namespace std;


namespace caffe {


template <typename Dtype>
void RoiGenLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    RoiGenParameter param = this->layer_param_.roi_gen_param();
    w_ = bottom[0]->width();
    h_ = bottom[0]->height();
    n_ = param.num_object();
    nw_ = param.net_w();
    nh_ = param.net_h();
    coords_ = param.num_coord();
    classes_ = param.num_class();
    prop_num_ = param.prop_num();

    box_num_ = 1;
    if(classes_ > 0)
      box_num_ = classes_;

    softmax_ = param.softmax();

    thresh_ = param.thresh();

    int anchor_x_size = param.anchor_x_size();
    int anchor_y_size = param.anchor_y_size();

    CHECK_EQ(anchor_x_size, anchor_y_size);
    CHECK_EQ(anchor_x_size, n_);
    
    vector<int> bias_shape;
    bias_shape.push_back(2*n_);
    biases_.Reshape(bias_shape);

    Dtype* l_biases = biases_.mutable_cpu_data();

    caffe_set(n_ * 2, Dtype(0.5), l_biases);

    for(int i=0; i<n_; i++)
    {
        l_biases[2*i + 0] = param.anchor_x(i);
        l_biases[2*i + 1] = param.anchor_y(i);
    }

    batch_ = bottom[0]->num();
    outputs_ = h_*w_*n_*(classes_ + coords_ + 1);
    inputs_ = outputs_;
    output_.ReshapeLike(*bottom[0]);
    CHECK_EQ(outputs_, bottom[0]->count(1));
}


template <typename Dtype>
void RoiGenLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(prop_num_);
  top[1]->Reshape(top_shape); // roi classification label
  top_shape.push_back(5);
  top[0]->Reshape(top_shape); // roi detected
}

template <typename Dtype>
void RoiGenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype* input_data = bottom[0]->cpu_data();
  Dtype* top_data_0 = top[0]->mutable_cpu_data();
  Dtype* top_data_1 = top[1]->mutable_cpu_data();

  //int size = coords_ + classes_ + 1;

  caffe_set(top[0]->count(), Dtype(0.), top_data_0);
  caffe_set(top[1]->count(), Dtype(0.), top_data_1);


  Dtype* l_output = output_.mutable_cpu_data();
  Dtype* l_biases = biases_.mutable_cpu_data();
  caffe_copy(bottom[0]->count(), input_data, l_output); 

  vector< vector<float> > pos_boxes;

  for (int b = 0; b < batch_; ++b){
    Dtype *pRes = l_output + b*outputs_;
    get_predict_boxes(b, w_, h_, l_biases, n_, classes_, pRes, pos_boxes, thresh_);
  }

  sort(pos_boxes.begin(), pos_boxes.end(), mycomp);

  int inserted = 0;
  for(; inserted < pos_boxes.size() && inserted < prop_num_; inserted++)
  {
    top_data_0[inserted*5 + 0] = pos_boxes[inserted][5];
    top_data_0[inserted*5 + 1] = pos_boxes[inserted][0] * nw_;
    top_data_0[inserted*5 + 2] = pos_boxes[inserted][1] * nh_;
    top_data_0[inserted*5 + 3] = pos_boxes[inserted][2] * nw_;
    top_data_0[inserted*5 + 4] = pos_boxes[inserted][3] * nh_;
    top_data_1[inserted] = 1;
  }

}

template <typename Dtype>
void RoiGenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_CLASS(RoiGenLayer);
REGISTER_LAYER_CLASS(RoiGen);

}  // namespace caffe
