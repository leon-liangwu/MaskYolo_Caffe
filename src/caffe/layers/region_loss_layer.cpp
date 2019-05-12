#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/box.hpp"
#include "opencv2/opencv.hpp"


namespace caffe {

static int data_seen = 0;

template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  RegionLossParameter param = this->layer_param_.region_loss_param();

  w_ = bottom[0]->width();
  h_ = bottom[0]->height();
  n_ = param.num_object();
  coords_ = param.num_coord();
  classes_ = param.num_class();
  display_inter_ = param.display_inter();

  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();

  softmax_ = param.softmax();

  thresh_ = param.thresh();
  with_mask_ = param.with_mask();
  with_rcnn_ = param.with_rcnn() || with_mask_;
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
  
  
  lab_count_ = bottom[1]->count(1);
  
  if(with_mask_)
  {
    mask_w_ = param.mask_w();
    mask_h_ = param.mask_h();
    truths_ = lab_count_ - mask_w_ * mask_h_;
    LOG(INFO)<<"mask w: "<<mask_w_<<" mask_h: "<<mask_h_<<" truths_: "<<truths_;
    label_stride_ = 6;
    max_box_num_ = truths_ / label_stride_;
  }
  else
  {
    truths_ = lab_count_;
    label_stride_ = 5;
    max_box_num_ = truths_ / label_stride_;
  }
  CHECK_EQ(0, truths_ % label_stride_);
  output_.ReshapeLike(*bottom[0]);
  CHECK_EQ(outputs_, bottom[0]->count(1));
  num_train_ = 0;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  output_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  data_seen += batch_;
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  int size = coords_ + classes_ + 1;

  
  /*
  for(int b=0; b<batch_; b++)
  {
    for(int i=0; i<max_box_num_;i++)
    {
        if(label_data[i*label_stride_+0+b*lab_count_] == 0) break;
        printf("%d %d %f %f %f %f %f %f\n", b, i, label_data[i*label_stride_+0+b*lab_count_], label_data[i*label_stride_+1+b*lab_count_], 
          label_data[i*label_stride_+2+b*lab_count_], label_data[i*label_stride_+3+b*lab_count_], label_data[i*label_stride_+4+b*lab_count_]
          , label_data[i*label_stride_+5+b*lab_count_]);
    }
  }
  */

  Dtype* diff = diff_.mutable_cpu_data();
  Dtype* l_biases = biases_.mutable_cpu_data();

  caffe_set(diff_.count(), Dtype(0.), diff);

  Dtype* l_output = output_.mutable_cpu_data();
  caffe_copy(diff_.count(), input_data, l_output); 

  Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0);
  Dtype avg_iou(0.0), recall(0.0), avg_cat(0.0), avg_obj(0.0), avg_anyobj(0.0);
  Dtype obj_count(0), class_count(0);

  int i,j,b,t,n;


  for (b = 0; b < batch_; ++b){
    for (j = 0; j < h_; ++j) {
      for (i = 0; i < w_; ++i) {
        for (n = 0; n < n_; ++n) {
          int index = (size * n + 4) * w_ * h_ + j * w_ + i + b*outputs_;
          l_output[index] = logistic_activate(l_output[index]);
        }
      }
    }
  }

  if (softmax_){
    for (b = 0; b < batch_; ++b){
      for (j = 0; j < h_; ++j) {
        for (i = 0; i < w_; ++i) {
          for (n = 0; n < n_; ++n) {
            int index = (size * n + 5) * w_ * h_ + j * w_ + i + b*outputs_;
            softmax(l_output + index, classes_, (Dtype)1.0, l_output + index);
          }
        }
      }
    }
  }
  cv::Mat mask_mat;
  for (b = 0; b < batch_; ++b) {
    /*
    if(with_mask_)
    {
      
       for(int i=0; i<max_box_num_;i++)
      {
          if(label_data[i*label_stride_+0+b*lab_count_] == 0) break;
          printf("%d %d %f %f %f %f %f %f\n", b, i, label_data[i*label_stride_+0+b*lab_count_], label_data[i*label_stride_+1+b*lab_count_], 
            label_data[i*label_stride_+2+b*lab_count_], label_data[i*label_stride_+3+b*lab_count_], label_data[i*label_stride_+4+b*lab_count_]
            , label_data[i*label_stride_+5+b*lab_count_]);
      }
      
      //cv::Mat mask_mat = cv::Mat::zeros(mask_h_,mask_w_, CV_8UC1);
      mask_mat = cv::Mat::zeros(mask_h_,mask_w_, CV_8UC1);
      for(int r=0; r<mask_h_; r++)
      {
        for(int c=0;c<mask_w_;c++)
        {
          mask_mat.at<uchar>(r,c) = (uchar)label_data[b*lab_count_ + truths_ + r * mask_w_ + c];
        }
      }
      cv::imshow("mask", mask_mat);
      cv::waitKey(0);
    }
    */
    for (j = 0; j < h_; ++j) {
      for (i = 0; i < w_; ++i) {
        
        if(with_mask_)
        {

          float y_shift = j+0.5;
          float x_shift = i+0.5;
          int x_mask = x_shift * mask_w_ / w_;
          int y_mask = y_shift * mask_h_ / h_;
          if(label_data[b*lab_count_ + truths_ + y_mask * mask_w_ + x_mask] == 255)
            continue;
        }
        
        for (n = 0; n < n_; ++n) {
          //int index = size*(j*w_*n_ + i*n_ + n) + b*outputs_;
          int index = size * n * w_ * h_ + j * w_ + i + b*outputs_;
          
          box pred = get_region_box(l_output, l_biases, n, index, i, j, w_, h_);
          float best_iou = 0;
          for(t = 0; t < max_box_num_; ++t){
            box truth = float_to_box(label_data+ t*label_stride_ + b*lab_count_);
            if(!truth.x) break;
            float iou = box_iou(pred, truth);
            if (iou > best_iou) {
                best_iou = iou;
            }
          }
          avg_anyobj += l_output[index + 4*w_*h_];
          
          if (best_iou > thresh_) {
            diff[index + 4*w_*h_] = 0;
          }
          else {
            noobj_loss += noobject_scale_ * pow(l_output[index + 4*w_*h_], 2);
            diff[index + 4*w_*h_] =  (-1) * noobject_scale_ * ((0 - l_output[index + 4*w_*h_]) * logistic_gradient(l_output[index + 4*w_*h_]));

            //noobj_loss += -noobject_scale_ * pow(l_output[index + 4], 2)*log(1-l_output[index + 4]+0.000001);
            //diff[index + 4] =  (-1) * noobject_scale_ * (2*l_output[index + 4]*log(1-l_output[index + 4]+0.000001) -
            //  pow(l_output[index + 4], 2)/(1-l_output[index + 4]+0.00001))* logistic_gradient(l_output[index + 4]);

          }
/*
          if(data_seen < 12800){
            box truth = {0};
            truth.x = (i + .5)/w_;
            truth.y = (j + .5)/h_;
            truth.w = l_biases[2*n]/w_;
            truth.h = l_biases[2*n+1]/h_;
            delta_region_box(truth, l_output, l_biases, n, index, i, j, w_, h_, diff, .01, coord_loss, area_loss);
          }
     */     
        }
      }
    }
    for(t = 0; t < max_box_num_; ++t) {
      box truth = float_to_box(label_data+ t*label_stride_ + b*lab_count_);
      if(!truth.x) break;
      /*
      cv::Point p1((int)((truth.x - truth.w/2) * mask_w_),(int)((truth.y - truth.h/2) * mask_h_));
      cv::Point p2((int)((truth.x + truth.w/2) * mask_w_),(int)((truth.y + truth.h/2) * mask_h_));
      cv::rectangle(mask_mat, p1, p2, cv::Scalar(255,255,255),1);
      cv::imshow("rect", mask_mat);
      cv::waitKey(0);
      */
      float best_iou = 0;
      int best_index = 0;
      int best_n = 0;
      i = (truth.x * w_);
      j = (truth.y * h_);
      box truth_shift = truth;
      truth_shift.x = 0;
      truth_shift.y = 0;
      for(n = 0; n < n_; ++n) {
          //int index = size*(j*w_*n_ + i*n_ + n) + b*outputs_;
          int index = size * n * w_ * h_ + j * w_ + i + b*outputs_;
          box pred = get_region_box(l_output, l_biases, n, index, i, j, w_, h_);

          pred.w = l_biases[2*n]/w_;
          pred.h = l_biases[2*n+1]/h_;
          
          pred.x = 0;
          pred.y = 0;
          float iou = box_iou(pred, truth_shift);
          if (iou > best_iou){
              best_index = index;
              best_iou = iou;
              best_n = n;
          }
      }

      float iou = delta_region_box(truth, l_output, l_biases, best_n, best_index, i, j, w_, h_, diff, coord_scale_, coord_loss, area_loss);
      if(iou > .5) recall += 1;
      avg_iou += iou;

      avg_obj += l_output[best_index + 4*w_*h_];

      obj_loss += object_scale_ * pow(1 - l_output[best_index + 4*w_*h_], 2);
      diff[best_index + 4*w_*h_] = (-1) * object_scale_ * (1 - l_output[best_index + 4*w_*h_]) * logistic_gradient(l_output[best_index + 4*w_*h_]);
      //obj_loss += -object_scale_ * pow(1 - l_output[best_index + 4], 2) * log(l_output[best_index + 4] + 0.000001);
      //diff[best_index + 4] = (-1) * object_scale_ * (2*(l_output[best_index + 4]-1)*log(l_output[best_index + 4]+0.000001) +
      //  pow(1 - l_output[best_index + 4], 2)/(l_output[best_index + 4]+0.000001)) * logistic_gradient(l_output[best_index + 4]);

      int class_ind = label_data[t*label_stride_ + b*lab_count_ + 4];
      delta_region_class(l_output, diff, best_index + 5*w_*h_, class_ind, classes_, class_scale_, avg_cat, class_loss, w_*h_);

      obj_count += 1;
      class_count += 1;
    }
  }

  obj_count += 0.01;
  class_count += 0.01;

  class_loss /= class_count;
  coord_loss /= obj_count;
  area_loss /= obj_count;
  obj_loss /= obj_count;
  noobj_loss /= (w_ * h_ * n_ * batch_ - obj_count);

  loss = class_loss + coord_loss + area_loss + obj_loss + noobj_loss;
  top[0]->mutable_cpu_data()[0] = loss;

  avg_iou /= obj_count;
  avg_cat /= class_count;
  avg_obj /= obj_count;
  avg_anyobj /= (w_*h_*n_*batch_ - obj_count);
  recall /= obj_count;
  obj_count /= batch_;

  if(display_inter_ > 0 && num_train_ % display_inter_ == 0)
  {
    LOG(INFO) << "loss: " << loss << " class_loss: " << class_loss << " obj_loss: " 
        << obj_loss << " noobj_loss: " << noobj_loss << " coord_loss: " << coord_loss
        << " area_loss: " << area_loss;
    LOG(INFO) << "avg_iou: " << avg_iou << " Class: " << avg_cat << " Obj: "
        << avg_obj << " No Obj: " << avg_anyobj << " Avg Recall: " << recall << " count: "<<(obj_count);
    num_train_ = 0;
  }
  num_train_++;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();

    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}  // namespace caffe
