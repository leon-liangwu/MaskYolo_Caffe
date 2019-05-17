#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/box.hpp"

namespace caffe {


template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  YoloLossParameter param = this->layer_param_.yolo_loss_param();

  w_ = bottom[0]->width();
  h_ = bottom[0]->height();
  nw_ = param.net_w();
  nh_ = param.net_h();
  total_ = param.num_object();
  n_ = total_;
  coords_ = param.num_coord();
  classes_ = param.num_class();

  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();

  ignore_thresh_ = param.ignore_thresh();
  truth_thresh_ = param.truth_thresh();

  int anchor_x_size = param.anchor_x_size();
  int anchor_y_size = param.anchor_y_size();

  CHECK_EQ(anchor_x_size, anchor_y_size);
  CHECK_EQ(anchor_x_size, total_);

  for(int i=0; i<total_; i++)
  {
      biases_.push_back(param.anchor_x(i));
      biases_.push_back(param.anchor_y(i));
  }

  anchor_mask_ = false;
  if(param.mask_size() > 0)
  {
    anchor_mask_ = true;
    n_ = param.mask_size();
    for(int i=0; i<n_; i++)
    {
      mask_.push_back(param.mask(i));
    }
  }

  batch_ = bottom[0]->num();
  outputs_ = h_*w_*n_*(classes_ + coords_ + 1);
  inputs_ = outputs_;
  truths_ = bottom[1]->count(1);
  max_box_num_ = truths_ / 5;
  output_.ReshapeLike(*bottom[0]);

  CHECK_EQ(outputs_, bottom[0]->count(1));
  CHECK_EQ(0, truths_ % 5);

  num_train_ = 0;

}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  output_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  int size = coords_ + classes_ + 1;

  /*
  for(int b=0; b<batch_; b++)
  {
    for(int i=0; i<max_box_num_;i++)
    {
        if(label_data[i*5+0+b*truths_] == 0) break;
        printf("%d %d %f %f %f %f %f\n", b, i, label_data[i*5+0+b*truths_], label_data[i*5+1+b*truths_], 
          label_data[i*5+2+b*truths_], label_data[i*5+3+b*truths_], label_data[i*5+4+b*truths_]);
    }
  }
  */

  Dtype* diff = diff_.mutable_cpu_data();

  caffe_set(diff_.count(), Dtype(0.), diff);

  Dtype* output_data = output_.mutable_cpu_data();
  caffe_copy(diff_.count(), input_data, output_data); 

  Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0);
  Dtype avg_iou(0.0), recall(0.0), avg_cat(0.0), avg_obj(0.0), avg_anyobj(0.0);
  Dtype obj_count(0), class_count(0);

  int i,j,b,t,n;

  for (b = 0; b < batch_; ++b){
    for (j = 0; j < h_; ++j) {
      for (i = 0; i < w_; ++i) {
        for (n = 0; n < n_; ++n) {
          for(int c = 0; c < 2; ++c)
          {
            int index = (size * n + c) * w_ * h_ + j * w_ + i + b*outputs_;
            output_data[index] = logistic_activate(output_data[index]);
          }
          for(int c = 0; c < classes_ + 1; ++c)
          {
            int index = (size * n + 4 + c) * w_ * h_ + j * w_ + i + b*outputs_;
            output_data[index] = logistic_activate(output_data[index]);
          }
        }
      }
    }
  }

  for (b = 0; b < batch_; ++b) {
    for (j = 0; j < h_; ++j) {
      for (i = 0; i < w_; ++i) {
        for (n = 0; n < n_; ++n) {
          int index = size * n * w_ * h_ + j * w_ + i + b*outputs_;
          box pred = get_yolo_box(output_data, biases_, mask_[n], index, i, j, w_, h_, nw_, nh_);
          float best_iou = 0;
          int best_t = 0;
          for(t = 0; t < max_box_num_; ++t){
            box truth = float_to_box(label_data+ t*5 + b*truths_);
            if(!truth.x) break;
            float iou = box_iou(pred, truth);
            if (iou > best_iou) {
                best_iou = iou;
                best_t = t;
            }
          }
          int obj_index = index + 4 * w_ * h_;
          avg_anyobj += output_data[obj_index];
          diff[obj_index] = (-1) * noobject_scale_ * (0 - output_data[obj_index]);
          if (best_iou > ignore_thresh_) {
            diff[obj_index] = 0;
          }
          else
          {
            noobj_loss += 1.0 * pow(output_data[obj_index], 2);
          }

          if (best_iou > truth_thresh_) {
            diff[obj_index] = (-1) * object_scale_ * (1- output_data[obj_index]);
            int class_ind = label_data[best_t*5 + b*truths_ + 4];
            int class_index = index + 5 * w_ * h_;
            obj_loss += 1.0 * pow(1 - output_data[obj_index], 2);
            delta_yolo_class(output_data, diff, class_index, class_ind, classes_, class_scale_, avg_cat, class_loss, w_*h_);
            box truth = float_to_box(label_data + best_t * 5 + b*truths_);
            delta_yolo_box(truth, output_data, biases_, mask_[n], obj_index, i, j, w_, h_, nw_, nh_, diff, coord_scale_ + 2 * (1-truth.w*truth.h), coord_loss, area_loss);
          }
        }
      }
    }
    for(t = 0; t < max_box_num_; ++t) {
      box truth = float_to_box(label_data+ t*5 + b*truths_);

      if(!truth.x) break;
      float best_iou = 0;
      int best_n = 0;
      i = (truth.x * w_);
      j = (truth.y * h_);
      box truth_shift = truth;
      truth_shift.x = 0;
      truth_shift.y = 0;
      for(n = 0; n < total_; ++n) {
          box pred = {0};
          pred.w = biases_[2*n]/nw_;
          pred.h = biases_[2*n+1]/nh_;
          float iou = box_iou(pred, truth_shift);
          if (iou > best_iou){
              best_iou = iou;
              best_n = n;
          }
      }

      int mask_n = int_index(mask_, best_n, n_);
      //LOG(INFO)<<"best n: "<<best_n<<" mask_n: "<<mask_n<<" mask_ :"<<mask_[0];
      if(mask_n >= 0) {
        
        int box_index = size * mask_n * w_ * h_ + j * w_ + i + b*outputs_;
        float iou = delta_yolo_box(truth, output_data, biases_, best_n, box_index, i, j, w_, h_, nw_, nh_, diff, coord_scale_+ 2*(1-truth.w*truth.h), coord_loss, area_loss);
  
        int obj_index = box_index + 4*w_*h_;

        avg_obj += output_data[obj_index];
        obj_loss += 1.0 * pow(1 - output_data[obj_index], 2);
        diff[obj_index] = (-1) * object_scale_ * (1 - output_data[obj_index]);

        int class_index = box_index + 5*w_*h_;
        int class_ind = label_data[t*5 + b*truths_ + 4];
        delta_yolo_class(output_data, diff, class_index, class_ind, classes_, class_scale_, avg_cat, class_loss, w_*h_);

        obj_count += 1;
        class_count += 1;
        if(iou > 0.5) recall += 1;
        avg_iou += iou;
        }
      
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


  if(num_train_ % 100 == 0)
  {
    LOG(INFO) << "loss_v3 "<<mask_[0]<<" - "<<mask_[n_-1]<<": " << loss << " class_loss: " << class_loss << " obj_loss: " 
          << obj_loss << " noobj_loss: " << noobj_loss << " coord_loss: " << coord_loss
          << " area_loss: " << area_loss;
    LOG(INFO) << "avg_iou: "<<mask_[0]<<" - "<<mask_[n_-1]<<": " << avg_iou << " Class: " << avg_cat << " Obj: "
          << avg_obj << " No Obj: " << avg_anyobj << " Avg Recall: " << recall << " count: "<<(obj_count);
    num_train_ = 0;
  }
  num_train_++;
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);

}  // namespace caffe
