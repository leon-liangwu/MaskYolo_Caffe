#include <functional>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/layers/decode_rois_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/box.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include<cstdlib>
#include<ctime>

using namespace cv;
using namespace std;

static float recall = 0;
static float gt_count = 0;
static int g_max_box_num = 30;
static int g_label_stride = 5;
static int save_ind = 0;


namespace caffe {



template <typename Dtype>
void saperate_pos_neg(const Dtype* label_data, vector< vector<float> > &pred_box, float thresh,
  vector< vector<float> > &pos_boxes, vector< vector<float> > &neg_boxes, int nw, int nh)
{
    
    vector<box> gt_boxes_v;
    for(int t = 0; t < g_max_box_num; ++t) {
      box truth = float_to_box(label_data+ t*g_label_stride);     
      if(!truth.x) break;
      gt_boxes_v.push_back(truth);
    }

    vector<box> pd_boxes_v;
    vector<box> pd_boxes_pos;
    vector<box> pd_boxes_neg;
    //cout<<"pred box num: "<<pred_box.size()<<endl;
    for(int i=0; i<pred_box.size(); i++)
    {
      float bf[4];
      bf[0] = pred_box[i][0], bf[1] = pred_box[i][1], bf[2] = pred_box[i][2], bf[3] = pred_box[i][3];
      box pred = float_to_box(bf);
      pd_boxes_v.push_back(pred);
    }

    

    for(int i=0; i< pred_box.size(); i++)
    {
      float x = pred_box[i][0];
      float y = pred_box[i][1];
      float w = pred_box[i][2];
      float h = pred_box[i][3];
      /*
      pred_box[i][0] = max(0.0f, (x - w/2) * side);
      pred_box[i][1] = max(0.0f, (y - h/2) * side);
      pred_box[i][2] = min(1.0f*(side-1), (x + w/2) * side);
      pred_box[i][3] = min(1.0f*(side-1), (y + h/2) * side);
      */
      pred_box[i][0] = max(0.0f, (x - w/2));
      pred_box[i][1] = max(0.0f, (y - h/2));
      pred_box[i][2] = min(1.0f, (x + w/2));
      pred_box[i][3] = min(1.0f, (y + h/2));
      float max_iou = 0;
      for(int j=0; j<gt_boxes_v.size(); j++)
      {
        float iou = box_iou(pd_boxes_v[i], gt_boxes_v[j]);
        //float iod = box_iod(pd_boxes_v[i], gt_boxes_v[j]);
        //float iol = box_iol(pd_boxes_v[i], gt_boxes_v[j]);
        //float ior = box_ior(pd_boxes_v[i], gt_boxes_v[j]);
        if(iou > max_iou)
        {
          
          max_iou = iou;
        }
      }
      pred_box[i][0] *= nw;
      pred_box[i][1] *= nh;
      pred_box[i][2] *= nw;
      pred_box[i][3] *= nh;
      if(max_iou > 0.5)
      {
        pos_boxes.push_back(pred_box[i]);
        pd_boxes_pos.push_back(pd_boxes_v[i]);
      }
      else
      {
        neg_boxes.push_back(pred_box[i]);
        pd_boxes_neg.push_back(pd_boxes_v[i]);
      }
    }
    
    /*
    cout<<"gt_size: "<<gt_boxes_v.size()<<" pos_size: "<<pd_boxes_pos.size()<<" neg_size: "<<pd_boxes_neg.size()<<endl;
    Mat img = Mat::zeros(500,500, CV_8UC3);
    draw_boxes(img, gt_boxes_v, Scalar(0,0,255));
    draw_boxes(img, pd_boxes_neg, Scalar(255,0,0));
    draw_boxes(img, pd_boxes_pos, Scalar(0,255,0));

    imshow("img", img);

    waitKey(0);
    */
    
}

template <typename Dtype>
void get_pos_neg_boxes(int batch_ind, int side_w, int side_h, Dtype *biases, int box_num, int cls_num, Dtype* pRes, const Dtype* truth, 
  vector< vector<float> > &pos_boxes, vector< vector<float> > &neg_boxes, float thresh, int nw, int nh)
{
    vector< vector<float> > pred_box;
    int locations = side_w * side_h;
    box *boxes = (box *)calloc(locations*box_num, sizeof(box));
    float **probs = (float **)calloc(locations*box_num, sizeof(float *));
    for(int j = 0; j < locations*box_num; ++j) probs[j] = (float *)calloc(cls_num+1, sizeof(float));
    get_region_boxes(side_w, side_h, biases, box_num, cls_num, pRes, thresh, probs, boxes, 1);
    do_nms_obj(boxes, probs, side_w*side_h*box_num, cls_num, 0.4);
    get_valid_boxes(batch_ind, pred_box, boxes, probs, side_w*side_h*box_num, cls_num);
    saperate_pos_neg(truth, pred_box, 0.1, pos_boxes, neg_boxes, nw, nh);
    for(int j = 0; j < locations*box_num; ++j) free(probs[j]);
    free(probs);
    free(boxes);
}


template <typename Dtype>
void insert_gt_boxes(int batch_ind, const Dtype* label_data, 
  vector< vector<float> > &gt_boxes)
{
  vector<box> gt_boxes_v;
  for(int t = 0; t < g_max_box_num; ++t) {
    box truth = float_to_box(label_data+ t*g_label_stride);
    if(!truth.x) break;
    gt_boxes_v.push_back(truth);
  }

  for(int i=0; i<gt_boxes_v.size(); i++)
  {
    vector<float> box_v;
    float x = gt_boxes_v[i].x;
    float y = gt_boxes_v[i].y;
    float w = gt_boxes_v[i].w;
    float h = gt_boxes_v[i].h;

    /*
    box_v.push_back(batch_ind);
    box_v.push_back(max(0.0f,(x-w/2) * side));
    box_v.push_back(max(0.0f,(y-h/2) * side));
    box_v.push_back(min(1.0f * (side-1),(x+w/2) * side));
    box_v.push_back(min(1.0f * (side-1),(y+h/2) * side));
    box_v.push_back(1.0);
    */

    box_v.push_back(batch_ind);
    box_v.push_back(max(0.0f,x-w/2));
    box_v.push_back(max(0.0f,y-h/2));
    box_v.push_back(min(1.0f,x+w/2));
    box_v.push_back(min(1.0f,y+h/2));
    box_v.push_back(1.0);
    gt_boxes.push_back(box_v);
  }

}

template <typename Dtype>
void random_neg_roi(Dtype* top, const Dtype* label_data, int batch_, int label_stride_, int truths_, int lab_count_, int max_box_num_, int net_w, int net_h)
{
  int float_acc = 100000;
  int rand_batch = 0;
  float rand_x = 0;
  float rand_y = 0;
  float rand_w = 0;
  float rand_h = 0;
  while(1)
  {
    srand((unsigned)time(NULL));
    rand_batch = rand() % batch_;
    rand_x = 1.0 * (rand() % float_acc ) / float_acc;
    rand_y = 1.0 * (rand() % float_acc ) / float_acc;
    float max_w = 2.0 * min(rand_x, 1.0f - rand_x);
    rand_w = 1.0 * (rand() % float_acc ) / float_acc * max_w;
    float max_h = 2.0 * min(rand_y, 1.0f - rand_y);
    rand_h = 1.0 * (rand() % float_acc ) / float_acc * max_h;

    box rand_box = {rand_x, rand_y, rand_w, rand_h};
    bool match = false;
    for(int i =0; i<max_box_num_; i++)
    {
      box truth_box = float_to_box(label_data + i*label_stride_ + rand_batch * lab_count_);
      if(truth_box.x == 0) break;
      float iou = box_iou(rand_box, truth_box);
      if(iou > 0.5)
      {
        match = true;
        break;
      }
    }
    if(match == false)
      break;
  }

  top[0] = rand_batch;
  top[1] = (rand_x - 0.5 * rand_w) * net_w;
  top[2] = (rand_y - 0.5 * rand_h) * net_h;
  top[3] = (rand_x + 0.5 * rand_w) * net_w;
  top[4] = (rand_y + 0.5 * rand_h) * net_h;
}

template <typename Dtype>
void target_transform(Dtype* top, box& truth, box& roi_box )
{

    //LOG(INFO)<<"truth: "<<truth.x<<" "<<truth.y<<" "<<truth.w<<" "<<truth.h;
    //LOG(INFO)<<"roibox: "<<roi_box.x<<" "<<roi_box.y<<" "<<roi_box.w<<" "<<roi_box.h;
    top[0] = (truth.x - roi_box.x) / roi_box.w;
    top[1] = (truth.y - roi_box.y) / roi_box.h;
    top[2] = log(truth.w / roi_box.w);
    top[3] = log(truth.h / roi_box.h);

}

/*
template <typename Dtype>
void get_roimask_gt(Dtype* mask_roi, int target_size, const Dtype* mask_lab, int mask_w, int mask_h, box& roi_box, int instance_id, int box_cls)
{
  float roi_x = (roi_box.x - roi_box.w / 2) * mask_w;
  float roi_y = (roi_box.y - roi_box.h / 2) * mask_h;
  float roi_w = roi_box.w * mask_w;
  float roi_h = roi_box.h * mask_h;

  for( int r = 0; r< target_size; r++)
  {
    for(int c = 0; c< target_size; c++)
    {
      int mask_r = 1.0 * roi_h / target_size * r + roi_y;
      int mask_c = 1.0 * roi_w / target_size * c + roi_x;
      mask_r = mask_r >= 0 ? mask_r : 0;
      mask_r = mask_r < mask_h ? mask_r : mask_h - 1;
      mask_c = mask_c >= 0 ? mask_c : 0;
      mask_c = mask_c < mask_w ? mask_c : mask_w - 1;
      if((int)mask_lab[mask_r * mask_w + mask_c] == instance_id * 20)
        mask_roi[r*target_size + c] = box_cls + 1;
    }
  }
}
*/

template <typename Dtype>
void get_roimask_gt(Dtype* mask_roi, int target_size, const Dtype* mask_lab, int mask_w, int mask_h, box& roi_box, int instance_id, int box_cls)
{
  float roi_x = (roi_box.x - roi_box.w / 2) * mask_w;
  float roi_y = (roi_box.y - roi_box.h / 2) * mask_h;
  float roi_w = roi_box.w * mask_w;
  float roi_h = roi_box.h * mask_h;

  cv::Rect rect((int)roi_x, (int)roi_y, (int)roi_w, (int)roi_h);
  cv::Mat mask_img = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);

  for( int r = 0; r< rect.height; r++)
  {
    for(int c = 0; c< rect.width; c++)
    {
      if((int)mask_lab[(r+rect.y) * mask_w + c + rect.x] == instance_id)
        mask_img.at<uchar>(r,c) = 255;
    }
  }
  //cv::imshow("mask roi", mask_img);
  cv::resize(mask_img, mask_img, cv::Size(target_size, target_size));

  for( int r = 0; r< target_size; r++)
  {
    for(int c = 0; c< target_size; c++)
    {
      mask_roi[r*target_size + c] = round(1.0 * mask_img.at<uchar>(r,c) / 255);
    }
  }
}

template <typename Dtype>
void get_roikps_gt(Dtype* kps_roi, int target_size, const Dtype* kps_lab, int mask_w, int mask_h, box& roi_box, int instance_id, int box_cls)
{
  int stride = 35;
  int index = 0;
  while(kps_lab[index*stride + 0]!=instance_id && kps_lab[index*stride + 0]!=0)
    index+=stride;

  if(0 == kps_lab[index*stride + 0])
    return;

  
  float roi_x = (roi_box.x - roi_box.w / 2) * mask_w;
  float roi_y = (roi_box.y - roi_box.h / 2) * mask_h;
  float roi_w = roi_box.w * mask_w;
  float roi_h = roi_box.h * mask_h;
  
  //cv::Rect rect((int)roi_x, (int)roi_y, (int)roi_w, (int)roi_h);
  //cv::Mat mask_img = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
  
  for(int i=0; i<17; i++)
  {
    if(kps_lab[index + 1 + i*2 + 0] == 0 && kps_lab[index + 1 + i*2 + 1] == 0) continue;
    float ox = kps_lab[index + 1 + i*2 + 0] * mask_w;
    float oy = kps_lab[index + 1 + i*2 + 1] * mask_h;

    float nx = (ox - roi_x);
    float ny = (oy - roi_y);
    
    int cx = round(nx * 1.0 * target_size/roi_w); 
    int cy = round(ny * 1.0 * target_size/roi_h); 

    if(cy < 0 || cy > (roi_y + roi_h)
      || cx < 0 || cx > (roi_x + roi_w))
      continue;
    

    //cv::circle(mask_img, cv::Point(int(nx), int(ny)), 2, cv::Scalar(255,255,255), 2);
    kps_roi[cy*target_size + cx] = i+1;
  }
  //cv::imshow("kps ori", mask_img);

}

template <typename Dtype>
void get_roikps_gaussian_gt(Dtype* kps_roi, int target_size, const Dtype* kps_lab, int mask_w, int mask_h, box& roi_box, int instance_id, int box_cls, float sigma)
{
  int stride = 35;
  int index = 0;
  while(kps_lab[index + 0]!=instance_id && kps_lab[index + 0]!=0)
  {
    index+=stride;
  }

  if(0 == kps_lab[index + 0])
    return;

  
  float roi_x = (roi_box.x - roi_box.w / 2) * mask_w;
  float roi_y = (roi_box.y - roi_box.h / 2) * mask_h;
  float roi_w = roi_box.w * mask_w;
  float roi_h = roi_box.h * mask_h;
  
  //cv::Rect rect((int)roi_x, (int)roi_y, (int)roi_w, (int)roi_h);
  //cv::Mat mask_img = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
  
  for(int i=0; i<17; i++)
  {
    if(kps_lab[index + 1 + i*2 + 0] == 0 && kps_lab[index + 1 + i*2 + 1] == 0) continue;
    float ox = kps_lab[index + 1 + i*2 + 0] * mask_w;
    float oy = kps_lab[index + 1 + i*2 + 1] * mask_h;

    float cx = ox - roi_x;
    float cy = oy - roi_y;
    
    if(cy < 0 || cy > (roi_y + roi_h)
      || cx < 0 || cx > (roi_x + roi_w))
      continue;
    //cv::circle(mask_img, cv::Point(int(cx), int(cy)), 2, cv::Scalar(255,255,255), 2);
    cx = cx * 1.0 * target_size/roi_w; 
    cy = cy * 1.0 * target_size/roi_h; 
    
    //kps_roi[ny*target_size + nx] = i+1;
    Dtype* entry = kps_roi + i*target_size*target_size; 
    for(int y=0; y<target_size; y++)
    {
      for(int x=0; x<target_size; x++)
      {
        float d2 = (x-cx)*(x-cx) + (y-cy)*(y-cy);
        float exponent = d2 / 2.0 / sigma / sigma;
        if(exponent > 4.6052){ //ln(100) = -ln(1%)
          continue;
        }
        entry[y * target_size + x] += exp(-exponent);
        if(entry[y * target_size + x] > 1) 
          entry[y * target_size + x] = 1;
      }
    }
  }
  //cv::imshow("roi kps ", mask_img);
  for(int y=0; y<target_size; y++)
  {
    for(int x=0; x<target_size; x++)
    {
      float maximum = 0;
      for(int i=0; i<17; i++)
      {
        float value = kps_roi[i*target_size*target_size + y*target_size + x];
        maximum = (maximum > value) ? maximum : value; 
      }
      kps_roi[17*target_size*target_size + y*target_size + x] = max(1.0-maximum, 0.0);
    }
  }
}



template <typename Dtype>
void DecodeRoisLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    DecodeRoisParameter param = this->layer_param_.decode_rois_param();
    w_ = bottom[0]->width();
    h_ = bottom[0]->height();
    n_ = param.num_object();
    nw_ = param.net_w();
    nh_ = param.net_h();
    coords_ = param.num_coord();
    classes_ = param.num_class();
    prop_num_ = param.prop_num();
    sigma_ = param.sigma();

    box_num_ = 1;
    if(classes_ > 0)
      box_num_ = classes_;

    softmax_ = param.softmax();

    thresh_ = param.thresh();
    target_size_ = param.target_size();
    with_mask_ = param.with_mask();
    with_kps_ = param.with_kps();

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
      mask_w_ = param.mask_w();
      mask_h_ = param.mask_h();
      truths_ = lab_count_ - mask_w_ * mask_h_;
      label_stride_ = 6;
      kps_stride_ = 35;
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

    g_max_box_num = max_box_num_;
    g_label_stride = label_stride_;
    CHECK_EQ(0, truths_ % label_stride_);
    output_.ReshapeLike(*bottom[0]);
    CHECK_EQ(outputs_, bottom[0]->count(1));
}


template <typename Dtype>
void DecodeRoisLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(prop_num_);
  top[1]->Reshape(top_shape); // roi classification label
  top_shape.push_back(5);
  top[0]->Reshape(top_shape); // roi detected

  vector<int> box_top_shape;
  box_top_shape.push_back(prop_num_);
  box_top_shape.push_back(box_num_ * 4);
  top[2]->Reshape(box_top_shape); // box groud truth 
  top[3]->Reshape(box_top_shape); // box inside weights
  top[4]->Reshape(box_top_shape); // box outside weights

  vector<int> mask_top_shape;
  mask_top_shape.push_back(prop_num_);
  mask_top_shape.push_back(target_size_);
  mask_top_shape.push_back(target_size_);
  top[5]->Reshape(mask_top_shape); // mask ground truth

  vector<int> kps_top_shape;
  kps_top_shape.push_back(prop_num_);
  if(sigma_ > 0)
  {
    kps_top_shape.push_back(18);
  }
  kps_top_shape.push_back(target_size_);
  kps_top_shape.push_back(target_size_);
  top[6]->Reshape(kps_top_shape); // mask ground truth
}

template <typename Dtype>
void DecodeRoisLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data_0 = top[0]->mutable_cpu_data();
  Dtype* top_data_1 = top[1]->mutable_cpu_data();
  Dtype* top_data_2 = top[2]->mutable_cpu_data();
  Dtype* top_data_3 = top[3]->mutable_cpu_data();
  Dtype* top_data_4 = top[4]->mutable_cpu_data();
  Dtype* top_data_5 = 0;
  Dtype* top_data_6 = 0;
  if(with_mask_)
    top_data_5 = top[5]->mutable_cpu_data();
  //int size = coords_ + classes_ + 1;

  if(with_kps_)
    top_data_6 = top[6]->mutable_cpu_data();
  //int size = coords_ + classes_ + 1;

  caffe_set(top[0]->count(), Dtype(0.), top_data_0);
  caffe_set(top[1]->count(), Dtype(0.), top_data_1);
  caffe_set(top[2]->count(), Dtype(0.), top_data_2);
  caffe_set(top[3]->count(), Dtype(0.), top_data_3);
  caffe_set(top[4]->count(), Dtype(0.), top_data_4);
  if(with_mask_)
    caffe_set(top[5]->count(), Dtype(0.), top_data_5);
  if(with_kps_)
    caffe_set(top[6]->count(), Dtype(0.), top_data_6);

  Dtype* l_output = output_.mutable_cpu_data();
  Dtype* l_biases = biases_.mutable_cpu_data();
  caffe_copy(bottom[0]->count(), input_data, l_output); 

  vector< vector<float> > pos_boxes, neg_boxes, gt_boxes;

  for (int b = 0; b < batch_; ++b){
    Dtype *pRes = l_output + b*outputs_;
    const Dtype *gt_label = label_data + b*lab_count_;
    insert_gt_boxes(b, gt_label, gt_boxes);
    get_pos_neg_boxes(b, w_, h_, l_biases, n_, classes_, pRes, gt_label, pos_boxes, neg_boxes, thresh_, nw_, nh_);
  }

  //if (phase_ == TRAIN)
    //pos_boxes.insert(pos_boxes.end(), gt_boxes.begin(),gt_boxes.end());
  recall += pos_boxes.size();
  gt_count += gt_boxes.size();
  //cout<<"gt num: "<<gt_boxes.size()<<" pos num: "<<pos_boxes.size()<<" neg num: "<<neg_boxes.size()<<" recall: "<<recall/gt_count<<endl;
  //sort(pos_boxes.begin(), pos_boxes.end(), mycomp);
  //sort(neg_boxes.begin(), neg_boxes.end(), mycomp);

  int inserted = 0;
  for(; inserted < pos_boxes.size() && inserted < prop_num_; inserted++)
  {
    top_data_0[inserted*5 + 0] = pos_boxes[inserted][5];
    top_data_0[inserted*5 + 1] = pos_boxes[inserted][0];
    top_data_0[inserted*5 + 2] = pos_boxes[inserted][1];
    top_data_0[inserted*5 + 3] = pos_boxes[inserted][2];
    top_data_0[inserted*5 + 4] = pos_boxes[inserted][3];
    top_data_1[inserted] = 1;
  }
  int recod_ind = inserted;
  for(; inserted - recod_ind < neg_boxes.size() && inserted < prop_num_; inserted++)
  {
    top_data_0[inserted*5 + 0] = neg_boxes[inserted - recod_ind][5];
    top_data_0[inserted*5 + 1] = neg_boxes[inserted - recod_ind][0];
    top_data_0[inserted*5 + 2] = neg_boxes[inserted - recod_ind][1];
    top_data_0[inserted*5 + 3] = neg_boxes[inserted - recod_ind][2];
    top_data_0[inserted*5 + 4] = neg_boxes[inserted - recod_ind][3];
    top_data_1[inserted] = 0;
  }
  recod_ind = inserted;
  for(; inserted < prop_num_; inserted++)
  {
    random_neg_roi(top_data_0+inserted*5, label_data, batch_, label_stride_, truths_, lab_count_, max_box_num_, nw_, nh_);
  }
  
  
  /* 
  cv::Mat mask_mat;
  for (int b = 0; b < batch_; ++b)
  {
    if(with_mask_)
    {
      
       for(int i=0; i<max_box_num_;i++)
      {
          if(label_data[i*label_stride_+0+b*lab_count_] == 0) break;
          printf("%d %d %f %f %f %f %f %f\n", b, i, label_data[i*label_stride_+0+b*lab_count_], label_data[i*label_stride_+1+b*lab_count_], 
            label_data[i*label_stride_+2+b*lab_count_], label_data[i*label_stride_+3+b*lab_count_], label_data[i*label_stride_+4+b*lab_count_], 
            label_data[i*label_stride_+5+b*lab_count_]);
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
      for(int t = 0; t < max_box_num_; ++t) {
      box truth = float_to_box(label_data+ t*label_stride_ + b*lab_count_);
      if(!truth.x) break;
      
      cv::Point p1((int)((truth.x - truth.w/2) * mask_w_),(int)((truth.y - truth.h/2) * mask_h_));
      cv::Point p2((int)((truth.x + truth.w/2) * mask_w_),(int)((truth.y + truth.h/2) * mask_h_));
      cv::rectangle(mask_mat, p1, p2, cv::Scalar(255,255,255),1);
      }
      //cv::imshow("mask", mask_mat);
      //cv::waitKey(0);
      char save_path[200];
      sprintf(save_path, "./debug/%05d.jpg", save_ind);
      cv::imwrite(save_path, mask_mat);
      save_ind++;
    }
  }
  
  */
    

  for(int i =0; i<prop_num_; i++)
  {
    //LOG(INFO)<<"top label "<< i <<" "<<top_data_1[i];
    if(top_data_1[i] == 0) continue;
    int batch_ind = top_data_0[i * 5 + 0];
    float x1 = (float)top_data_0[i * 5 + 1] / nw_;
    float y1 = (float)top_data_0[i * 5 + 2] / nh_;
    float x2 = (float)top_data_0[i * 5 + 3] / nw_;
    float y2 = (float)top_data_0[i * 5 + 4] / nh_;
    box roi_box = {(x1+x2)/2, (y1+y2)/2, x2 - x1, y2 - y1};
    int best_ind = 0;
    float best_iou = 0;
    for(int t = 0; t<max_box_num_; t++)
    {
      box truth_box = float_to_box(label_data+ t*label_stride_ + batch_ind*lab_count_);
      float iou = box_iou(roi_box, truth_box);
      if(iou > best_iou)
      {
        best_iou = iou;
        best_ind = t;
      }
    }
    int box_cls = label_data[ best_ind * label_stride_ + batch_ind * lab_count_ + 4];

    //box target transform
    box truth_box = float_to_box(label_data + best_ind * label_stride_ + batch_ind * lab_count_);
    target_transform(top_data_2 + box_num_ * 4 * i + box_cls * 4, truth_box, roi_box );

    /*
    //box ground truth
    top_data_2[box_num_ * 4 * i + box_cls * 4 + 0 ] = label_data[ best_ind * label_stride_ + batch_ind * lab_count_ + 0];
    top_data_2[box_num_ * 4 * i + box_cls * 4 + 1 ] = label_data[ best_ind * label_stride_ + batch_ind * lab_count_ + 1];
    top_data_2[box_num_ * 4 * i + box_cls * 4 + 2 ] = label_data[ best_ind * label_stride_ + batch_ind * lab_count_ + 2];
    top_data_2[box_num_ * 4 * i + box_cls * 4 + 3 ] = label_data[ best_ind * label_stride_ + batch_ind * lab_count_ + 3];
    */

    //box inside weights
    top_data_3[box_num_ * 4 * i + box_cls * 4 + 0 ] = 1.0;
    top_data_3[box_num_ * 4 * i + box_cls * 4 + 1 ] = 1.0;
    top_data_3[box_num_ * 4 * i + box_cls * 4 + 2 ] = 1.0;
    top_data_3[box_num_ * 4 * i + box_cls * 4 + 3 ] = 1.0;

    //box outside weights
    top_data_4[box_num_ * 4 * i + box_cls * 4 + 0 ] = 1.0;
    top_data_4[box_num_ * 4 * i + box_cls * 4 + 1 ] = 1.0;
    top_data_4[box_num_ * 4 * i + box_cls * 4 + 2 ] = 1.0;
    top_data_4[box_num_ * 4 * i + box_cls * 4 + 3 ] = 1.0;

    //mask ground truth
    int instance_id = label_data[ best_ind * label_stride_ + batch_ind * lab_count_ + 5];
    if(with_mask_)
    {
      get_roimask_gt(top_data_5 + target_size_*target_size_*i, target_size_, 
        label_data+batch_ind * lab_count_+truths_, mask_w_, mask_h_, roi_box, instance_id, box_cls);

      /*
      LOG(INFO)<<" instance id: "<<instance_id;
      cv::Mat roi_mask_mat = cv::Mat::zeros(target_size_, target_size_, CV_8UC1);
      for(int r=0;r<target_size_;r++)
      {
        for(int c=0;c<target_size_;c++)
        {
          roi_mask_mat.at<uchar>(r,c) = (uchar) top_data_5[target_size_*target_size_*i + r*target_size_ + c] * 255;
        }
      }

      //cv::imshow("mask roi", roi_mask_mat);
      //cv::waitKey(0);
      char save_path[200];
      sprintf(save_path, "./debug/%05d.jpg", save_ind);
      cv::imwrite(save_path, roi_mask_mat);
      save_ind++;
      */
      
    }
    if(with_kps_)
    {
      int kps_ind = batch_ind * lab_count_ + truths_ + mask_w_ * mask_h_;
      if(sigma_ < 0.1)
      {
        get_roikps_gt(top_data_6 + target_size_*target_size_*i, target_size_, 
        label_data+kps_ind, mask_w_, mask_h_, roi_box, instance_id, box_cls);

        /*
        cv::Mat img_bac = cv::Mat::zeros(target_size_, target_size_, CV_8UC1);
        for(int r=0; r<target_size_; r++)
        {
          for(int c=0; c<target_size_; c++)
          {
            img_bac.at<uchar>(r, c) = top_data_6[target_size_*target_size_*i + r*target_size_ + c] * 255;
          }
        }
        cv::resize(img_bac, img_bac, cv::Size(128, 128));
        cv::imshow("back g", img_bac);
        cv::waitKey(0);
        */
      }
      else
      {
        get_roikps_gaussian_gt(top_data_6 + target_size_*target_size_*i*18, target_size_, 
        label_data+kps_ind, mask_w_, mask_h_, roi_box, instance_id, box_cls, sigma_);

        /*
        LOG(INFO)<<"instance id : "<<instance_id;
        const Dtype* lab_p = label_data+kps_ind;
        for(int ii=0;ii<3;ii++)
        {
          for(int kk=0; kk<35; kk++)
          {
            std::cout<<lab_p[ii*35 + kk]<<" ";
          }
          std::cout<<std::endl;

        }
        std::cout<<"====================="<<std::endl;
        std::cout<<std::endl;
        
        
        cv::Mat img_bac = cv::Mat::zeros(target_size_, target_size_, CV_8UC1);
        cv::Mat img_left = cv::Mat::zeros(target_size_, target_size_, CV_8UC1);
        cv::Mat img_right = cv::Mat::zeros(target_size_, target_size_, CV_8UC1);
        for(int r=0; r<target_size_; r++)
        {
          for(int c=0; c<target_size_; c++)
          {
            img_bac.at<uchar>(r, c) = top_data_6[target_size_*target_size_*(i*18+17) + r*target_size_ + c] * 255;
            img_left.at<uchar>(r, c) = top_data_6[target_size_*target_size_*(i*18+11) + r*target_size_ + c] * 255;
            img_right.at<uchar>(r, c) = top_data_6[target_size_*target_size_*(i*18+12) + r*target_size_ + c] * 255;
          }
        }
        cv::resize(img_bac, img_bac, cv::Size(128, 128));
        cv::resize(img_left, img_left, cv::Size(128, 128));
        cv::resize(img_right, img_right, cv::Size(128, 128));
        cv::imshow("back g", img_bac);
        cv::imshow("left", img_left);
        cv::imshow("right", img_right);
        cv::waitKey(0);
        */
      }
      
      
    }
  }

  /*
  for(int b = 0; b<batch_; b++)
  {
    Mat img = Mat::zeros(mask_h_, mask_w_, CV_8UC3);
    vector<box> gt_boxes_v;
    for(int t =0; t<max_box_num_; t++)
    {
      box truth_box = float_to_box(label_data + label_stride_ * t + b * lab_count_);
      if(truth_box.x == 0) break;
      gt_boxes_v.push_back(truth_box);
    }
    vector<box> pd_boxes_pos, pd_boxes_neg;
    for(int t=0; t<prop_num_; t++)
    {
      if(top_data_0[t*5+0] != b) continue;
      float x1 = top_data_0[t*5+1];
      float y1 = top_data_0[t*5+2];
      float x2 = top_data_0[t*5+3];
      float y2 = top_data_0[t*5+4];
      box pd_box = {(x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1)};
      if(top_data_1[t] == 0)
        pd_boxes_neg.push_back(pd_box);
      else
        pd_boxes_pos.push_back(pd_box);

    }
    for(int r = 0; r<mask_h_;r++)
    {
      for(int c= 0; c<mask_w_; c++)
      {
        img.at<uchar>(r,c*3+0) = (uchar)label_data[b*lab_count_ + truths_ + r * mask_w_ + c];
        img.at<uchar>(r,c*3+1) = (uchar)label_data[b*lab_count_ + truths_ + r * mask_w_ + c];
        img.at<uchar>(r,c*3+2) = (uchar)label_data[b*lab_count_ + truths_ + r * mask_w_ + c];
      }
    }
    //draw_boxes(img, gt_boxes_v, Scalar(0,0,255));
    //draw_boxes(img, pd_boxes_neg, Scalar(255,0,0));
    draw_boxes(img, pd_boxes_pos, Scalar(0,255,0));
    imshow("img", img);

    for(int t = 0; t<prop_num_; t++)
    {
      if(top_data_0[t*5+0] != b || top_data_1[t] == 0) continue;
      cv::Mat roi_mask_mat = cv::Mat::zeros(target_size_, target_size_, CV_8UC1);
      for(int r=0;r<target_size_;r++)
      {
        for(int c=0;c<target_size_;c++)
        {
          roi_mask_mat.at<uchar>(r,c) = (uchar) top_data_5[target_size_*target_size_*t + r*target_size_ + c] * 255;
        }
      }
      cv::imshow("mask roi", roi_mask_mat);
      cv::waitKey(0);
    }

    waitKey(0);
  }
  */
  
}

template <typename Dtype>
void DecodeRoisLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_CLASS(DecodeRoisLayer);
REGISTER_LAYER_CLASS(DecodeRois);

}  // namespace caffe
