#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using namespace cv;

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
Dtype constrain(Dtype min, Dtype max, Dtype a)
{
  if (a < min) return min;
  if (a > max) return max;
  return a;
}

float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)(rand()%100)/100 * (max - min)) + min;
}

template<typename Dtype>
Dtype rand_scale(Dtype s)
{
    Dtype scale = rand_uniform(1., s);
    if(rand()%2) return scale;
    return 1./scale;
}

template<typename Dtype>
void scale_image_channel(cv::Mat &im, int c, Dtype v)
{
    int i, j;
    for(j = 0; j < im.rows; ++j){
        for(i = 0; i < im.cols; ++i){
            im.at<float>(j,3*i+c) = v * im.at<float>(j,3*i+c);
        }
    }
}

void constrain_image(cv::Mat &im)
{
    int i, j, c;
    for(i = 0; i < im.rows; i++)
        for(j=0; j<im.cols; j++)
            for(c=0; c<3; c++)
            {
                if(im.at<float>(i, 3*j+c) < 0) im.at<float>(i, 3*j+c) = 0;
                if(im.at<float>(i, 3*j+c) > 1) im.at<float>(i, 3*j+c) = 1;
            }
}

template<typename Dtype>
void distort_image(cv::Mat &im, Dtype hue, Dtype sat, Dtype val)
{
    cv::Mat im_hsv;
    cv::Mat im_scale;
    Dtype scale = 1.0 /255;
    im.convertTo(im_scale, CV_32F, scale);
    cvtColor(im_scale, im_hsv, COLOR_BGR2HSV);
    scale_image_channel(im_hsv, 1, sat);
    scale_image_channel(im_hsv, 2, val);
    int i, j;
    for(i=0; i<im_hsv.rows; i++)
        for(j=0; j<im_hsv.cols; j++)
        {
            im_hsv.at<float>(i,3*j) = im_hsv.at<float>(i,3*j) + hue;
            if(im_hsv.at<float>(i,3*j) > 360) im_hsv.at<float>(i,3*j) -= 360;
            if(im_hsv.at<float>(i,3*j) <  0 ) im_hsv.at<float>(i,3*j) += 360;
        }
    cvtColor(im_hsv, im_scale, COLOR_HSV2BGR);
    constrain_image(im_scale);
    im_scale = im_scale *255;
    im_scale.convertTo(im, CV_8U);
}

template<typename Dtype>
void random_distort_image(cv::Mat &img, Dtype hue, Dtype saturation, Dtype exposure)
{
    Dtype dhue = rand_uniform(-hue, hue);
    Dtype dsat = rand_scale(saturation);
    Dtype dexp = rand_scale(exposure);
    distort_image(img, dhue, dsat, dexp);
}

void crop_image(cv::Mat &img_org, cv::Mat &img_crop, int dx, int dy, int w, int h)
{
    img_crop =  cv::Mat(h, w, img_org.type(), Scalar::all(128));
    int im_c = img_org.channels();
    int im_h = img_org.rows;
    int im_w = img_org.cols;
    for(int k = 0; k < im_c; ++k){
        for(int j = 0; j < h; ++j){
            for(int i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                if(r != constrain(0, im_h-1, r) || c != constrain(0, im_w-1, c)) {
                  //std::cout<<int(img_crop.at<uchar>(j, i * im_c + k))<<std::endl;
                  continue;
                }
                //c = constrain(0, im_w-1, c);
                img_crop.at<uchar>(j, i * im_c + k) = img_org.at<uchar>(r, c * im_c + k);
            }
        }
    }
}

void pad_image(cv::Mat &img_crop, vector<BoxLabel>& box_labels, float ratio)
{
    int ow = img_crop.cols;
    int oh = img_crop.rows;
    float src_ratio = 1.0 * ow / oh;
    if(src_ratio < ratio + 0.001 && src_ratio > ratio - 0.001) return;

    if(src_ratio > ratio)
    {
      int sy = (ow / ratio - oh) / 2;
      cv::Mat img_pad =  cv::Mat(ow  / ratio, ow, img_crop.type(), cv::Scalar::all(128)); 
      cv::Mat roi_pad = img_pad(cv::Rect(0, sy, ow, oh));
      img_crop.copyTo(roi_pad);
      for(int i =0 ; i< box_labels.size(); i++)
      {
        float y = box_labels[i].box_[1];
        float h = box_labels[i].box_[3];
        box_labels[i].box_[1] = (y * oh + sy) / ow;
        box_labels[i].box_[3] = h * oh / ow;
      }
      img_crop = img_pad.clone();
    }
    else
    {
      int sx = (oh * ratio - ow) / 2;
      cv::Mat img_pad =  cv::Mat(oh, oh * ratio, img_crop.type(), cv::Scalar::all(128));
      cv::Mat roi_pad = img_pad(cv::Rect(sx, 0, ow, oh));
      img_crop.copyTo(roi_pad);
      for(int i =0 ; i< box_labels.size(); i++)
      {
        float x = box_labels[i].box_[0];
        float w = box_labels[i].box_[2];
        box_labels[i].box_[0] = (x * ow + sx) / oh;
        box_labels[i].box_[2] = w * ow / oh;
      }
      img_crop = img_pad.clone();
    }

}

void correct_boxes(vector<BoxLabel> &ori_labels, vector<BoxLabel>& box_labels, 
                    float dx, float dy, float sx, float sy, bool flip)
{
    for(int i =0; i<ori_labels.size(); i++)
    {
        float *box   = ori_labels[i].box_;

        float left   = box[0] - box[2]/2;
        float right  = box[0] + box[2]/2;
        float top    = box[1] - box[3]/2;
        float bottom = box[1] + box[3]/2; 

        left   = left  * sx - dx;
        right  = right * sx - dx;
        top    = top   * sy - dy;
        bottom = bottom* sy - dy;

        if(flip){
            float swap = left;
            left = 1. - right;
            right = 1. - swap;
        }

        left   = constrain(0.f, 1.f, left);
        right  = constrain(0.f, 1.f, right);
        top    = constrain(0.f, 1.f, top);
        bottom = constrain(0.f, 1.f, bottom);

        BoxLabel boxlabel;
        boxlabel.box_[0] = (left+right)/2;
        boxlabel.box_[1] = (top+bottom)/2;
        boxlabel.box_[2] = (right - left);
        boxlabel.box_[3] = (bottom - top);
        boxlabel.class_label_ = ori_labels[i].class_label_;
        boxlabel.box_index_ = ori_labels[i].box_index_;

        boxlabel.box_[2] = constrain(0.f, 1.f, boxlabel.box_[2]);
        boxlabel.box_[3] = constrain(0.f, 1.f, boxlabel.box_[3]);

        box_labels.push_back(boxlabel);
    }
}

void pad_image(cv::Mat &img_crop, vector<BoxLabel>& box_labels)
{
    int ow = img_crop.cols;
    int oh = img_crop.rows;
    if(ow == oh) return;

    if(ow > oh)
    {
      int sy = (ow - oh) / 2;
      cv::Mat img_pad =  cv::Mat(ow, ow, img_crop.type(), Scalar::all(128)); 
      cv::Mat roi_pad = img_pad(cv::Rect(0, sy, ow, oh));
      img_crop.copyTo(roi_pad);
      for(int i =0 ; i< box_labels.size(); i++)
      {
        float y = box_labels[i].box_[1];
        float h = box_labels[i].box_[3];
        box_labels[i].box_[1] = (y * oh + sy) / ow;
        box_labels[i].box_[3] = h * oh / ow;
      }
      img_crop = img_pad.clone();
    }
    else
    {
      int sx = (oh - ow) / 2;
      cv::Mat img_pad =  cv::Mat(oh, oh, img_crop.type(), Scalar::all(128)); 
      cv::Mat roi_pad = img_pad(cv::Rect(sx, 0, ow, oh));
      img_crop.copyTo(roi_pad);
      for(int i =0 ; i< box_labels.size(); i++)
      {
        float x = box_labels[i].box_[0];
        float w = box_labels[i].box_[2];
        box_labels[i].box_[0] = (x * ow + sx) / oh;
        box_labels[i].box_[2] = w * ow / oh;
      }
      img_crop = img_pad.clone();
    }

}


void draw_box(cv::Mat img, vector<BoxLabel> & box_vec)
{
  int img_w = img.cols;
  int img_h = img.rows;
  for(int i=0; i<box_vec.size(); i++)
  {
    float *box = box_vec[i].box_;
    int left   = (box[0] - box[2]/2) * img_w;
    int top    = (box[1] - box[3]/2) * img_h;
    int right  = (box[0] + box[2]/2) * img_w;
    int bottom = (box[1] + box[3]/2) * img_h;

    cv::Point p1(left, top);
    cv::Point p2(right, bottom);

    cv::rectangle(img, p1, p2, Scalar(255, 0, 0), 5 );
  }
}

void place_image(cv::Mat &img_scale,float dx, float dy, cv::Mat &img_crop)
{
   int stt_x = std::max(0, int(dx));
   int stt_y = std::max(0, int(dy));
   int end_x = std::min(int(dx) + img_scale.cols, img_crop.cols);
   int end_y = std::min(int(dy) + img_scale.rows, img_crop.rows);

   for(int r = stt_y; r<end_y; r++)
   {
      for(int c = stt_x; c<end_x; c++)
      {
        int y = r - dy;
        int x = c - dx;
        if(img_scale.channels() == 3)
        {
          img_crop.at<uchar>(r, c*3 + 0) = img_scale.at<uchar>(y, x*3 + 0);
          img_crop.at<uchar>(r, c*3 + 1) = img_scale.at<uchar>(y, x*3 + 1);
          img_crop.at<uchar>(r, c*3 + 2) = img_scale.at<uchar>(y, x*3 + 2);
        }
        else
        {
          img_crop.at<uchar>(r, c) = img_scale.at<uchar>(y, x);
        }
        
      }
   }

}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob,
                                       vector<BoxLabel>& box_labels) {
  const int resize_w = param_.resize_w();
  const int resize_h = param_.resize_h();
  const float jitter = param_.jitter();
  const float scale_margin = param_.scale_margin();
  const bool keep_ratio = param_.keep_ratio();

  int float_size = datum.float_data_size();

  float wh_rate = 1.0;
  if(float_size % 5 == 1)
    wh_rate = datum.float_data(float_size-1);

  CHECK_EQ(((float_size-1) % 5) || (float_size % 5), true) <<
    "Every box label has 5 labels (class,  box), optionally plus a ratio value";

  vector<BoxLabel> ori_labels;
  for (int j = 0; j < float_size-1; j += 5) {
    BoxLabel box_label;
    box_label.class_label_ = datum.float_data(j);
    for (int k = 1; k < 5; ++k) {
      box_label.box_[k-1] = datum.float_data(j+k);
    }
    ori_labels.push_back(box_label);
  }

  // If datum is encoded, decoded and transform the cv::image.
  CHECK(datum.encoded()) << "For box data, datum must be encoded";
  CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
  cv::Mat cv_img;
  if (param_.force_color() || param_.force_gray()) {
  // If force_color then decode in color otherwise decode in gray.
    cv_img = DecodeDatumToCVMat(datum, param_.force_color());
  } else {
    cv_img = DecodeDatumToCVMatNative(datum);
  }

  bool mirror = Rand(2);
  bool distort = Rand(2);


  cv::Mat img_crop;

  if(keep_ratio)
  {
    int oh = cv_img.rows;
    int ow = oh * wh_rate;
    cv::resize(cv_img, cv_img, cv::Size(ow, oh));

    int dw = (ow*jitter);
    int dh = (oh*jitter);

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  ow - pleft - pright;
    int sheight = oh - ptop - pbot;

    float sx = (float)swidth  / ow;
    float sy = (float)sheight / oh;

    crop_image(cv_img, img_crop, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/ow)/sx;
    float dy = ((float)ptop /oh)/sy;

    correct_boxes(ori_labels, box_labels, dx, dy, 1./sx, 1./sy, mirror);

    float des_ratio = 1.0;
    if(resize_h > 0 && resize_w > 0)
    {
      des_ratio = 1.0 * resize_w / resize_h;
    }
    else
    {
      des_ratio = 1.0 * cv_img.cols / cv_img.rows;
    }

    pad_image(img_crop, box_labels, des_ratio);
  }
  else
  {
    int ow = cv_img.cols;
    int oh = cv_img.rows;
    img_crop =  cv::Mat(oh, ow, cv_img.type(), cv::Scalar::all(128));
    
    float dw = jitter * ow;
    float dh = jitter * oh;

    float new_ar = (ow + rand_uniform(-dw, dw)) / (oh + rand_uniform(-dh, dh));
    float scale = rand_uniform(1. - scale_margin, 1. + scale_margin);

    float nw, nh;
    float w = ow, h = oh;

    if(new_ar < 1){
        nh = scale * h;
        nw = nh * new_ar;
    } else {
        nw = scale * w;
        nh = nw / new_ar;
    }
    cv::Mat img_scale;
    cv::resize(cv_img, img_scale, cv::Size(nw, nh));

    float dx = rand_uniform(0, w - nw);
    float dy = rand_uniform(0, h - nh);

    place_image(img_scale, dx, dy, img_crop);
    correct_boxes(ori_labels, box_labels, -dx/w, -dy/h, nw/w, nh/h, mirror);
  }
  
  if (mirror) { 
     cv::flip(img_crop, img_crop, 1); // horizen flip
  }

  if (resize_w > 0 && resize_h > 0) {
    cv::resize(img_crop, img_crop, cv::Size(resize_w, resize_h));
  }

  if (distort) {
    random_distort_image(img_crop, 360.0*0.1, 1.5, 1.5); // hsv distortion
  }

  // Transform the cv::image into blob.
  Transform(img_crop, transformed_blob);

  /*
  int h_big = 224;
  int w_big = h_big * wh_rate;
  cv::resize(cv_img, cv_img, cv::Size(w_big, h_big));
  draw_box(cv_img, ori_labels);
  cv::imshow("img_org", cv_img);
  draw_box(img_crop, box_labels);
  cv::imshow("img_crop", img_crop);
  cv::waitKey(0);
  */
  
  return;
}

void mask_resize(cv::Mat& mask, cv::Mat& mask_scale, cv::Size ns)
{
  cv::Mat n_mask = cv::Mat::zeros(ns, CV_8UC1);
  for(int r = 0; r<ns.height; r++)
  {
    for(int c = 0; c<ns.width; c++)
    {
      int o_r = (int)(1.0 * mask.rows / n_mask.rows * r);
      int o_c = (int)(1.0 * mask.cols / n_mask.cols * c);
      n_mask.at<uchar>(r,c) = mask.at<uchar>(o_r, o_c);
    }
  }
  mask_scale = n_mask.clone();
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob,
                                       vector<BoxLabel>& box_labels,
                                       cv::Mat& mask) {
  const int resize_w = param_.resize_w();
  const int resize_h = param_.resize_h();
  const float jitter = param_.jitter();
  const float scale_margin = param_.scale_margin();
  const bool rotate = param_.rotate();
  const float rotate_angle = param_.rotate_angle();

  int float_size = datum.float_data_size();


  /*
  float wh_rate = 1.0;
  if(float_size % 6 == 1)
    wh_rate = datum.float_data(float_size-1);
  */
  CHECK_EQ(((float_size-1) % 6) || (float_size % 6), true) <<
    "Mask box label has 6 labels (class,  box), optionally plus a ratio value";

  vector<BoxLabel> ori_labels;
  for (int j = 0; j+6 < float_size-1; j += 6) {
    BoxLabel box_label;
    box_label.class_label_ = datum.float_data(j);
    int k = 1;
    for(k = 1; k < 5; ++k) {
      box_label.box_[k-1] = datum.float_data(j+k);
    }
    box_label.box_index_ = datum.float_data(j+k);
    ori_labels.push_back(box_label);
  }

  int mask_valid = (int)datum.float_data(float_size - 2);

  cv::Mat cv_img;
  DatumToImage_Mask(datum, cv_img, mask);

  bool mirror = Rand(2);
  bool distort = Rand(2);

  if(mask_valid && rotate)// && Rand(2))
  {
    float angle = rand_uniform(-rotate_angle, rotate_angle);
        //指定旋转中心  
    cv::Point2f center(cv_img.rows / 2., cv_img.cols / 2.);  
      
    //获取旋转矩阵（2x3矩阵）  
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);  
    float cos_a = rot_mat.at<double>(0, 0);
    float sin_a = rot_mat.at<double>(0, 1);

    //根据旋转矩阵进行仿射变换  
    cv::warpAffine(cv_img, cv_img, rot_mat, cv_img.size()); 
    cv::warpAffine(mask, mask, rot_mat, mask.size()); 


    std::cout<<cos_a<<" sin a: "<<sin_a<<std::endl;

    int rows = cv_img.rows;
    int cols = cv_img.cols;
  
    for(int i =0; i<ori_labels.size(); i++)
    {
      std::cout<<"index: "<<i<<std::endl;
      float x = ori_labels[i].box_[0];
      float y = ori_labels[i].box_[1];
      float w = ori_labels[i].box_[2];
      float h = ori_labels[i].box_[3];
      std::cout<<ori_labels[i].box_[0]<<" "<<ori_labels[i].box_[1]<<" "<<ori_labels[i].box_[2]<<" "<<ori_labels[i].box_[3]<<std::endl;
      ori_labels[i].box_[0] = (x-0.5)*cos_a + (y-0.5) * sin_a + 0.5;
      ori_labels[i].box_[1] = -(x-0.5)*sin_a + (y-0.5) * cos_a + 0.5;
      ori_labels[i].box_[2] = w;//w * cos_a + h * sin_a;
      ori_labels[i].box_[3] = h;//h * cos_a + w * sin_a;
      std::cout<<ori_labels[i].box_[0]<<" "<<ori_labels[i].box_[1]<<" "<<ori_labels[i].box_[2]<<" "<<ori_labels[i].box_[3]<<std::endl;
      float left = cols;
      float top = rows;
      float right = 0;
      float bottom = 0;
      int box_ind = ori_labels[i].box_index_;
      for(int r=0; r<rows; r++)
      {
        for(int c=0; c<cols; c++)
        {
          if(mask.at<uchar>(r, c) != box_ind)
            continue;

          left = min(left, (float)c);
          right = max(right, (float)c);
          top = min(top, (float)r);
          bottom = max(bottom, (float)r);
        }
      }
      ori_labels[i].box_[0] = (left + right) / 2 / cols;
      ori_labels[i].box_[1] = (top + bottom) / 2 / rows;
      ori_labels[i].box_[2] = (right - left) / 2 / cols;
      ori_labels[i].box_[3] = (bottom - top) / 2 / rows;
    }   

    //显示旋转效果  
    draw_box(cv_img, ori_labels);
    cv::imshow("rot_img", cv_img);  
    cv::imshow("rot_mask", mask*20); 
    
    cv::waitKey(0);

  }


  cv::Mat img_crop, mask_crop;

  int ow = cv_img.cols;
  int oh = cv_img.rows;
  img_crop =  cv::Mat(oh, ow, cv_img.type(), cv::Scalar::all(128));
  mask_crop =  cv::Mat::zeros(oh, ow, mask.type());
  
  float dw = jitter * ow;
  float dh = jitter * oh;

  float new_ar = (ow + rand_uniform(-dw, dw)) / (oh + rand_uniform(-dh, dh));
  float scale = rand_uniform(1. - scale_margin, 1. + scale_margin);

  float nw, nh;
  float w = ow, h = oh;

  if(new_ar < 1){
      nh = scale * h;
      nw = nh * new_ar;
  } else {
      nw = scale * w;
      nh = nw / new_ar;
  }
  cv::Mat img_scale, mask_scale;
  cv::resize(cv_img, img_scale, cv::Size(nw, nh));
  mask_resize(mask, mask_scale, cv::Size(nw, nh));

  float dx = rand_uniform(0, w - nw);
  float dy = rand_uniform(0, h - nh);

  place_image(img_scale, dx, dy, img_crop);
  place_image(mask_scale, dx, dy, mask_crop);
  correct_boxes(ori_labels, box_labels, -dx/w, -dy/h, nw/w, nh/h, mirror);

  
  if (mirror) { 
     cv::flip(img_crop, img_crop, 1); // horizen flip
     cv::flip(mask_crop, mask_crop, 1); // horizen flip
  }

  if (resize_w > 0 && resize_h > 0) {
    cv::resize(img_crop, img_crop, cv::Size(resize_w, resize_h));
    mask_resize(mask_crop, mask_crop, cv::Size(resize_w, resize_h));
     //mask_resize(mask, mask, cv::Size(resize_w, resize_h));
     cv::resize(mask, mask, cv::Size(resize_w, resize_h));
  }

  if (distort) {
    random_distort_image(img_crop, 360.0*0.1, 1.5, 1.5); // hsv distortion
  }

  mask = mask_crop.clone();

  // Transform the cv::image into blob.
  Transform(img_crop, transformed_blob);
  /*
  //draw_box(cv_img, ori_labels);
  draw_box(mask_crop, box_labels);
  //draw_box(mask, ori_labels);
  //cv::imshow("img_org", cv_img);
  cv::imshow("mask_crop", mask_crop);
  //cv::imshow("mask", mask);
  draw_box(img_crop, box_labels);
  cv::imshow("img_crop", img_crop);
  cv::waitKey(0);
  */
  return;
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {

  // If datum is encoded, decode and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int resize_w = param_.resize_w();
  const int resize_h = param_.resize_h();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = 3;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  if (resize_w > 0 && resize_h > 0) {
    shape[2] = resize_h;
    shape[3] = resize_w;
  }
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::  InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  const int resize_w = param_.resize_w();
  const int resize_h = param_.resize_h();
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  if (resize_w > 0 && resize_h > 0) {
    shape[2] = resize_h;
    shape[3] = resize_w;
  }
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  //const bool needs_rand = param_.mirror() ||
  //    (phase_ == TRAIN && param_.crop_size());
  const bool needs_rand = param_.mirror() || phase_ == TRAIN;
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
