#include "caffe/util/box.hpp"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>



namespace caffe {


//template <typename Dtype>
box float_to_box(const double *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}

box float_to_box(const float *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}




dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}


float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}


float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}


float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}


float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}


float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) + 
                pow(a.y-b.y, 2) + 
                pow(a.w-b.w, 2) + 
                pow(a.h-b.h, 2));
}


dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}


dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}


void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}


void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}


void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}


dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}

typedef struct{
    int index;
    int class_ind;
    float **probs;
} sortable_bbox;


int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class_ind] - b.probs[b.index][b.class_ind];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}


void do_nms_obj(box *boxes, float **probs, int total, int class_indes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *)calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class_ind = class_indes;
        s[i].probs = probs;
    }

    qsort(s, total, sizeof(sortable_bbox), nms_comparator);
    for(i = 0; i < total; ++i){
        if(probs[s[i].index][class_indes] == 0) continue;
        box a = boxes[s[i].index];
        for(j = i+1; j < total; ++j){
            box b = boxes[s[j].index];
            if (box_iou(a, b) > thresh){
                for(k = 0; k < class_indes+1; ++k){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}


void do_nms_sort(box *boxes, float **probs, int total, int class_indes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *)calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class_ind = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < class_indes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class_ind = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}


void do_nms(box *boxes, float **probs, int total, int class_indes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < class_indes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < class_indes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}

int int_index(vector<int>& a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        if(a[i] == val) return i;
    }
    return -1;
}


template <typename Dtype>
void flatten(Dtype *x, int size, int layers, int batch, int forward)
{
    Dtype *swap = (Dtype*) calloc(size*layers*batch, sizeof(Dtype));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(Dtype));
    free(swap);
}







int entry_index(int side_w, int side_h, int coords, int classes, int location, int entry)
{
    int n =   location / (side_w*side_h);
    int loc = location % (side_w*side_h);
    return n*side_w*side_h*(coords+classes+1) + entry*side_w*side_h + loc;
}



template <typename Dtype>
int max_index(Dtype *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

void get_valid_boxes(int batch_ind, vector< vector<float> > &pred_box, box *boxes, float **probs, int num, int classes)
{
    for(int i = 0; i < num; i++)
    {
        int class_ind = std::max(max_index(probs[i], classes), 0);

        float prob = probs[i][class_ind];
        //cout<<"class_ind: "<<class_ind<<" "<<"valid prob: "<<probs[i][0]<<endl;
        if(prob > 0)
        {
            vector<float> box_v;
            box_v.push_back(boxes[i].x);
            box_v.push_back(boxes[i].y);
            box_v.push_back(boxes[i].w);
            box_v.push_back(boxes[i].h);
            box_v.push_back(prob);
            box_v.push_back(batch_ind);
            pred_box.push_back(box_v);
            //cout<<"box: "<<boxes[i].x<<" "<<boxes[i].y<<" "<<boxes[i].h<<" "<<boxes[i].w<<" "<<prob<<endl;

        }
    }
}

void draw_boxes(cv::Mat &img, vector<box> &boxs_v, cv::Scalar color)
{
  for(int i=0; i<boxs_v.size(); i++)
  {
    cv::Point p1((boxs_v[i].x - boxs_v[i].w/2) * img.cols, (boxs_v[i].y - boxs_v[i].h/2)*img.rows);
    cv::Point p2((boxs_v[i].x + boxs_v[i].w/2) * img.cols, (boxs_v[i].y + boxs_v[i].h/2)*img.rows);
    cv::rectangle(img, p1, p2, color, 1);
  }
}


void get_predict_boxes(int batch_ind, vector< vector<float> > &pred_box, box *boxes, float **probs, int num, int classes)
{
    for(int i = 0; i < num; i++)
    {
        int class_ind = std::max(max_index(probs[i], classes), 0);

        float prob = probs[i][class_ind];
        //cout<<"class_ind: "<<class_ind<<" "<<"valid prob: "<<probs[i][0]<<endl;
        if(prob > 0)
        {
            float x = boxes[i].x;
            float y = boxes[i].y;
            float w = boxes[i].w;
            float h = boxes[i].h;
            vector<float> box_v;
            box_v.push_back(max(0.0f, (x - w/2)));
            box_v.push_back(max(0.0f, (y - h/2)));
            box_v.push_back(min(1.0f, (x + w/2)));
            box_v.push_back(min(1.0f, (y + h/2)));
            box_v.push_back(prob);
            box_v.push_back(batch_ind);
            pred_box.push_back(box_v);
            //cout<<"box: "<<boxes[i].x<<" "<<boxes[i].y<<" "<<boxes[i].h<<" "<<boxes[i].w<<" "<<prob<<endl;

        }
    }
}


    /*
    struct compclass {
    template <typename Dtype> bool operator() (vector<Dtype> &a,vector<Dtype> &b) { return (a[4]<b[4]);}
    } mycomp;
    */

    bool mycomp(vector<float> &a,vector<float> &b)
    {
        return (a[4]<b[4]);
    }

}
