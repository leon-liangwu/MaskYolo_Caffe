#ifndef BOX_H
#define BOX_H
#endif


#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include <cfloat>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;


namespace caffe {

typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

//template <typename Dtype>
box float_to_box(const double *f);

box float_to_box(const float *f);

float box_iou(box a, box b);


float box_rmse(box a, box b);

dbox diou(box a, box b);


void do_nms(box *boxes, float **probs, int total, int classes, float thresh);


void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);


void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);


box decode_box(box b, box anchor);


box encode_box(box b, box anchor);


// region loss functions
template <typename Dtype>
static inline Dtype logistic_activate(Dtype x){return 1./(1. + exp(-x));}

template <typename Dtype>
static inline Dtype logistic_gradient(Dtype x){return (1-x)*x;}


template <typename Dtype>
void flatten(Dtype *x, int size, int layers, int batch, int forward);

template <typename Dtype>
void softmax(Dtype *input, int n, Dtype temp, Dtype *output)
{
    int i;
    Dtype sum = 0;
    Dtype largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        Dtype e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}


template <typename Dtype>
box get_region_box(Dtype *x, Dtype *biases, int n, int index, int i, int j, int w, int h)
{
    int stride = w * h;
    box b;
    b.x = (i + logistic_activate(x[index + 0 * stride])) / w;
    b.y = (j + logistic_activate(x[index + 1 * stride])) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3 * stride]) * biases[2*n+1] / h;
    return b;
}

template <typename Dtype>
float delta_region_box(box truth, Dtype *x, Dtype *biases, int n, int index, int i, int j, int w, int h, Dtype *delta, float scale, Dtype &coord_loss, Dtype &area_loss)
{
    int stride = w * h;
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    Dtype tx = (truth.x*w - i);
    Dtype ty = (truth.y*h - j);
    Dtype tw = log(truth.w*w / biases[2*n]);
    Dtype th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0 * stride] = (-1) * scale * (tx - logistic_activate(x[index + 0 * stride])) * logistic_gradient(logistic_activate(x[index + 0 * stride]));
    delta[index + 1 * stride] = (-1) * scale * (ty - logistic_activate(x[index + 1 * stride])) * logistic_gradient(logistic_activate(x[index + 1 * stride]));
    delta[index + 2 * stride] = (-1) * scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = (-1) * scale * (th - x[index + 3 * stride]);


//    std::cout<<"delta coord: "<<delta[index + 0]<<" "<<delta[index + 1]<<" "<<delta[index + 2]<<" "<<delta[index + 3]<<std::endl;

    coord_loss += scale * (pow((float)((tx-logistic_activate(x[index + 0 * stride]))), 2) + pow((float)((ty - logistic_activate(x[index + 1 * stride]))), 2));
    area_loss += scale * (pow((float)((tw - x[index + 2 * stride])), 2) + pow((float)((th - x[index + 3 * stride])), 2));
    return iou;
}


template <typename Dtype>
void delta_region_class(Dtype *output, Dtype *delta, int index, int class_ind, int classes, float scale, Dtype &avg_cat, Dtype &class_loss, int stride)
{
    int n;
    
    for(n = 0; n < classes; ++n){
        delta[index + n * stride] = (-1) * scale * (((n == class_ind)?1 : 0) - output[index + n * stride]);
        class_loss += scale * pow((float)((((n == class_ind)?1 : 0) - output[index + n * stride])), 2);
        if(n == class_ind) avg_cat += output[index + n * stride];

    }
    
}

int entry_index(int side_w, int side_h, int coords, int classes, int location, int entry);

template <typename Dtype>
void get_region_boxes(int side_w, int side_h, Dtype *biases, int box_num, int cls_num, Dtype* pRes, float thresh, float** probs, box *boxes, int only_objectness)
{
    int locations = side_w * side_h;
    for (int i = 0; i < locations; ++i){
        int row = i / side_w;
        int col = i % side_w;
        for(int n = 0; n < box_num; ++n){
            int index = n*locations + i;
            for(int j = 0; j < cls_num; ++j){
                probs[index][j] = 0;
            }
            int obj_index  = entry_index(side_w, side_h, 4, 0, n*locations + i, 4);
            int box_index  = entry_index(side_w, side_h, 4, 0, n*locations + i, 0);
            float scale = logistic_activate(pRes[obj_index]) ;
            probs[index][0] = scale > thresh ? scale : 0;
            if(scale > thresh)
              boxes[index] = get_region_box(pRes, biases, n, box_index, col, row, side_w, side_h);
        }
    }
}

template <typename Dtype>
int max_index(Dtype *a, int n);

void get_valid_boxes(int batch_ind, vector< vector<float> > &pred_box, box *boxes, float **probs, int num, int classes);

void draw_boxes(cv::Mat &img, vector<box> &boxs_v, cv::Scalar color);

void get_predict_boxes(int batch_ind, vector< vector<float> > &pred_box, box *boxes, float **probs, int num, int classes);


template <typename Dtype>
void get_predict_boxes(int batch_ind, int side_w, int side_h, Dtype *biases, int box_num, int cls_num, Dtype* pRes,
  vector< vector<float> > &pred_boxes, float thresh)
{
    int locations = side_w * side_h;
    box *boxes = (box *)calloc(locations*box_num, sizeof(box));
    float **probs = (float **)calloc(locations*box_num, sizeof(float *));
    for(int j = 0; j < locations*box_num; ++j) probs[j] = (float *)calloc(cls_num+1, sizeof(float));
    get_region_boxes(side_w, side_h, biases, box_num, cls_num, pRes, thresh, probs, boxes, 1);
    do_nms_obj(boxes, probs, side_w*side_h*box_num, cls_num, 0.4);
    get_predict_boxes(batch_ind, pred_boxes, boxes, probs, side_w*side_h*box_num, cls_num);
    for(int j = 0; j < locations*box_num; ++j) free(probs[j]);
    free(probs);
    free(boxes);
}

bool mycomp(vector<float> &a,vector<float> &b);

}

