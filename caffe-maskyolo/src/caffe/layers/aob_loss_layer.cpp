#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/aob_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/box.hpp"

namespace caffe {


template <typename Dtype>
void AobLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  AobLossParameter param = this->layer_param_.aob_loss_param();

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
  ignore_w_ = param.ignore_w();
  ignore_h_ = param.ignore_h();
  truth_thresh_ = param.truth_thresh();

  with_mask_ = param.with_mask();
  with_kps_ = param.with_kps();
  with_rcnn_ = param.with_rcnn() || with_mask_;

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

  lab_count_ = bottom[1]->count(1);
  if(with_mask_ && !with_kps_)
  {
    mask_w_ = param.mask_w();
    mask_h_ = param.mask_h();
    truths_ = lab_count_ - mask_w_ * mask_h_;
    LOG(INFO)<<"mask w: "<<mask_w_<<" mask_h_: "<<mask_h_<<" truths_: "<<truths_;
    label_stride_ = 6;
    max_box_num_ = truths_ / label_stride_;
  }
  else if(with_mask_ && with_kps_)
  {
    mask_w_ = nw_;
    mask_h_ = nh_;
    truths_ = lab_count_ - mask_w_ * mask_h_;
    label_stride_ = 6;
    int kps_stride_ = 35;
    max_box_num_ = truths_ / (label_stride_ + kps_stride_);
    truths_ = truths_ - kps_stride_ * max_box_num_;
    LOG(INFO)<<"total label: "<<lab_count_<<" mask w: "<<mask_w_<<" mask_h_: "<<mask_h_<<" truths_: "<<truths_;    
  }
  else
  {
    truths_ = lab_count_;
    label_stride_ = 5;
    max_box_num_ = truths_ / label_stride_;
  }

  output_.ReshapeLike(*bottom[0]);
  CHECK_EQ(outputs_, bottom[0]->count(1));
  CHECK_EQ(0, truths_ % label_stride_);

  num_train_ = 0;

}

template <typename Dtype>
void AobLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  output_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void AobLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  int size = coords_ + classes_ + 1;

  /*
  for(int b=0; b<batch_; b++)
  {
    for(int i=0; i<max_box_num_;i++)
    {
        if(label_data[i*label_stride_+0+b*lab_count_] == 0) break;
        printf("%d %d %f %f %f %f %f\n", b, i, label_data[i*label_stride_+0+b*lab_count_], 
        label_data[i*label_stride_+1+b*lab_count_], label_data[i*label_stride_+2+b*lab_count_], 
        label_data[i*label_stride_+3+b*lab_count_], label_data[i*label_stride_+4+b*lab_count_]);
    }
  }
  */

  Dtype* diff = diff_.mutable_cpu_data();

  caffe_set(diff_.count(), Dtype(0.), diff);

  Dtype* output_data = output_.mutable_cpu_data();
  caffe_copy(diff_.count(), input_data, output_data); 

  Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), iou_loss(0.0);
  Dtype avg_iou(0.0), recall(0.0), avg_cat(0.0), avg_obj(0.0), avg_anyobj(0.0);
  Dtype obj_count(0), class_count(0), no_obj_count(0);

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
          
          int index = size * n * w_ * h_ + j * w_ + i + b*outputs_;
          box pred = get_yolo_box(output_data, biases_, mask_[n], index, i, j, w_, h_, nw_, nh_);
          float best_iou = 0;
          int best_t = 0;
          for(t = 0; t < max_box_num_; ++t){
            box truth = float_to_box(label_data+ t*label_stride_ + b*lab_count_);
            if(!truth.x) break;
            //if(biases_[2*n] < 0.05) continue;
            float iou = box_iou(pred, truth);
            if (iou > best_iou) {
                best_iou = iou;
                best_t = t;
            }
          }
          int obj_index = index + 4 * w_ * h_;
          diff[obj_index] = (-1) * noobject_scale_ * (0 - output_data[obj_index]);
          if (best_iou > ignore_thresh_ || output_data[obj_index] < 0.01) {
            diff[obj_index] = 0;
          }
          else
          {
            avg_anyobj += output_data[obj_index];
            noobj_loss += 1.0 * pow(output_data[obj_index], 2);
            no_obj_count += 1;
          }
          
          /*
          int index = size * n * w_ * h_ + j * w_ + i + b*outputs_;
          int obj_index = index + 4 * w_ * h_;
          bool intruth = false;
          for(t = 0; t < max_box_num_; ++t){
            box truth = float_to_box(label_data+ t*label_stride_ + b*lab_count_);
            if(!truth.x) break;
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            int best_n = 0;
            float best_iou = 0;
            for(int nn = 0; nn < total_; ++nn) {
                box pred = {0};
                pred.w = biases_[2*nn]/nw_;
                pred.h = biases_[2*nn+1]/nh_;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = nn;
                }
            }
            if(best_n == n && 
               (i+0.5)/w_ > truth.x - truth.w/2 &&
               (i+0.5)/w_ < truth.x + truth.w/2 &&
               (j+0.5)/h_ > truth.y - truth.h/2 &&
               (j+0.5)/h_ < truth.y + truth.h/2)
            {
              intruth = true;
            }
          }
          if (intruth || output_data[obj_index] < 0.01) {
            diff[obj_index] = 0;
          }
          else
          {
            diff[obj_index] = (-1) * noobject_scale_ * (0 - output_data[obj_index]);
            avg_anyobj += output_data[obj_index];
            noobj_loss += 1.0 * pow(output_data[obj_index], 2);
            no_obj_count += 1;
          }
          */

          
          if (best_iou > truth_thresh_) {
            diff[obj_index] = (-1) * object_scale_ * (1- output_data[obj_index]);
            int class_ind = label_data[best_t*label_stride_ + b*lab_count_ + 4];
            int class_index = index + label_stride_ * w_ * h_;
            obj_loss += 1.0 * pow(1 - output_data[obj_index], 2);
            delta_yolo_class(output_data, diff, class_index, class_ind, classes_, class_scale_, avg_cat, class_loss, w_*h_);
            box truth = float_to_box(label_data + best_t * label_stride_ + b*lab_count_);
            //delta_yolo_box(truth, output_data, biases_, mask_[n], obj_index, i, j, w_, h_, nw_, nh_, diff, coord_scale_ + 2 * (1-truth.w*truth.h), coord_loss, area_loss);
            diff_iou_loss2(truth, output_data, biases_, mask_[n], obj_index, i, j, w_, h_, diff, .01, iou_loss, nw_, nh_);
          }
          
        }
      }
    }
    for(t = 0; t < max_box_num_; ++t) {
      box truth = float_to_box(label_data+ t*label_stride_ + b*lab_count_);

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
        int obj_index = box_index + 4*w_*h_;
        //float iou = delta_yolo_box(truth, output_data, biases_, best_n, box_index, i, j, w_, h_, nw_, nh_, diff, coord_scale_+ 2*(1-truth.w*truth.h), coord_loss, area_loss);
        float iou = diff_iou_loss2(truth, output_data, biases_, best_n, box_index, i, j, w_, h_, diff, coord_scale_, iou_loss, nw_, nh_);
  

        avg_obj += output_data[obj_index];
        obj_loss += 1.0 * pow(1 - output_data[obj_index], 2);
        diff[obj_index] = (-1) * object_scale_ * (1 - output_data[obj_index]);

        int class_index = box_index + label_stride_*w_*h_;
        int class_ind = label_data[t*label_stride_ + b*lab_count_ + 4];
        delta_yolo_class(output_data, diff, class_index, class_ind, classes_, class_scale_, avg_cat, class_loss, w_*h_);

        if(truth.x < ignore_w_ && truth.y < ignore_h_) {
          diff[obj_index] = 0;
        }
        else {
          obj_count += 1;
          class_count += 1;
          if(iou > 0.5) recall += 1;
          avg_iou += iou;
        }
      }
      
    }
  }

  obj_count += 0.01;
  no_obj_count += 0.01;
  class_count += 0.01;

  class_loss /= class_count;
  iou_loss /= obj_count;
  obj_loss /= obj_count;
  noobj_loss /= no_obj_count;

  loss = class_loss + iou_loss + obj_loss + noobj_loss;
  top[0]->mutable_cpu_data()[0] = loss;

  avg_iou /= obj_count;
  avg_cat /= class_count;
  avg_obj /= obj_count;
  avg_anyobj /= no_obj_count;
  recall /= obj_count;
  obj_count /= batch_;

  if(!with_rcnn_)
  {
    if(num_train_ % 100 == 0)
    {
      LOG(INFO) << "loss_v3 "<<mask_[0]<<" - "<<mask_[n_-1]<<": " << loss << " class_loss: " << class_loss << " obj_loss: " 
            << obj_loss << " noobj_loss: " << noobj_loss << " iou_loss: " << iou_loss;
      LOG(INFO) << "avg_iou: "<<mask_[0]<<" - "<<mask_[n_-1]<<": " << avg_iou << " Class: " << avg_cat << " Obj: "
            << avg_obj << " No Obj: " << avg_anyobj << " Avg Recall: " << recall << " count: "<<(obj_count);
      num_train_ = 0;
    }
    num_train_++;
  }
}

template <typename Dtype>
void AobLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_CLASS(AobLossLayer);
REGISTER_LAYER_CLASS(AobLoss);

}  // namespace caffe
