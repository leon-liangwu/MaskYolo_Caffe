#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mask_box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
MaskBoxDataLayer<Dtype>::MaskBoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
MaskBoxDataLayer<Dtype>::~MaskBoxDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void MaskBoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  min_size_and = this->layer_param_.data_param().min_size_and();
  min_size_or = this->layer_param_.data_param().min_size_or();
  max_box_num = this->layer_param_.data_param().max_box_num();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
      vector<int> label_shape(1, batch_size);
      label_shape.push_back(max_box_num*6 + top_shape[2] * top_shape[3]);
      top[1]->Reshape(label_shape);
      for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
      }
  }
}

template <typename Dtype>
bool MaskBoxDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void MaskBoxDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void MaskBoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  Datum datum;
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);

      vector<int> label_shape(1, batch_size);
      label_shape.push_back(max_box_num*6 + top_shape[2] * top_shape[3]);
      batch->label_.Reshape(label_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    
    vector<BoxLabel> box_labels;
    cv::Mat mask; 
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_), box_labels, mask);
  
    // Copy label.
    if (this->output_labels_) {
      int label_offset = batch->label_.offset(item_id);
      transform_label_mask(top_label+label_offset, box_labels, mask);
      /*
      for(int i=0; i<max_box_num;i++)
      {
          if(top_label[i*5+0] == 0) break;
          printf("%d %f %f %f %f %f\n", i, top_label[i*5+0], top_label[i*5+1], 
            top_label[i*5+2], top_label[i*5+3], top_label[i*5+4]);
      }
      */
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void MaskBoxDataLayer<Dtype>::transform_label_mask(Dtype* top_label, const vector<BoxLabel>& box_labels, const cv::Mat& mask) {

  caffe_set(max_box_num * 6 + mask.rows * mask.cols, Dtype(0), top_label);
  //LOG(INFO)<<"box_labels size: "<<box_labels.size();
  
  for (int i = 0, j=0; i < box_labels.size() && j < max_box_num; ++i) {
    float x = box_labels[i].box_[0];
    float y = box_labels[i].box_[1];
    float w = box_labels[i].box_[2];
    float h = box_labels[i].box_[3];
    float id = box_labels[i].class_label_; 
    float ind = box_labels[i].box_index_;
    

    if(w < min_size_and && h < min_size_and) continue;

    if(w < min_size_or || h < min_size_or) continue;

    //LOG(INFO)<<"X Y W H: "<<x<<" "<<y<<" "<<w<<" "<<h<<" "<<id;

    top_label[j*6 + 0] = x;
    top_label[j*6 + 1] = y;
    top_label[j*6 + 2] = w;
    top_label[j*6 + 3] = h;
    top_label[j*6 + 4] = id;
    top_label[j*6 + 5] = ind;
    j++;
  }

  for( int r = 0; r < mask.rows; r++)
  {
    for( int c = 0; c<mask.cols; c++)
    {
      top_label[max_box_num*6 + r*mask.cols + c] = (float)mask.at<uchar>(r, c);
    }
  }
}

INSTANTIATE_CLASS(MaskBoxDataLayer);
REGISTER_LAYER_CLASS(MaskBoxData);

}  // namespace caffe
