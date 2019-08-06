#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void kernel_ResizeBlob(const int nthreads,const int num,const int channels, const Dtype* src, const int src_height, const int src_width,
			Dtype* dst, const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int i = index %( dst_height * dst_width);
			int c = (index/(dst_height * dst_width))%channels;
			int n = (index/(dst_height * dst_width))/channels;
			int src_offset = (n * channels + c) * src_height * src_width;
			int dst_offset = (n * channels + c) * dst_height * dst_width;

			const Dtype* src_data = src+src_offset;
			Dtype* dst_data = dst+dst_offset;

			int dst_h = i /dst_width;
			Dtype fh = dst_h * scale_h;
			const int src_h = floor(fh);
			fh -= src_h;
			const Dtype w_h0 = std::abs(1.0f - fh);
			const Dtype w_h1 = std::abs(fh);

			const int dst_offset_1 =  dst_h * dst_width;
			const int src_offset_1 =  src_h * src_width;

			int dst_w = i %dst_width;
			Dtype fw = dst_w * scale_w;
			const int src_w = floor(fw);
			fw -= src_w;
			const Dtype w_w0 = std::abs(1.0f - fw);
			const Dtype w_w1 = std::abs(fw);

			const int dst_idx = dst_offset_1 + dst_w;


			const int src_idx = src_offset_1 + src_w;
			Dtype res = (w_h0 * w_w0 * src_data[src_idx]);

		/*if (src_w + 1 < src_width)
			res += (w_h0 * w_w1 * src_data[src_idx + 1]);
		if (src_h + 1 < src_height)
			res += (w_h1 * w_w0 * src_data[src_idx + src_width]);
		if (src_w + 1 < src_width && src_h + 1 < src_height)
			res += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);*/

		if (src_w < src_width)
			res += (w_h0 * w_w1 * src_data[src_idx]);
		if (src_h < src_height)
			res += (w_h1 * w_w0 * src_data[src_idx]);
		if (src_w < src_width && src_h < src_height)
			res += (w_h1 * w_w1 * src_data[src_idx]);

			dst_data[dst_idx] = res;
		}
	}

  template <typename Dtype>
  void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {

	CHECK(bottom[0]->num() == top[0]->num())<<"bottom[0]->num() == top[0]->num()";
	CHECK(bottom[0]->channels() == top[0]->channels())<< "bottom[0]->channels() == top[0]->channels()";

	const int src_num = bottom[0]->num();
	const int src_channels = bottom[0]->channels();
	const int src_height = bottom[0]->height();
	const int src_width = bottom[0]->width();

	const int dst_channels = top[0]->channels();
	const int dst_height = top[0]->height();
	const int dst_width = top[0]->width();

	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	int loop_n = dst_height * dst_width*dst_channels*src_num;
	const Dtype* src_data = bottom[0]->gpu_data();
	Dtype* dst_data = top[0]->mutable_gpu_data();
	kernel_ResizeBlob<Dtype> << <CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >> >(
		loop_n,src_num,src_channels,
		src_data, src_height,src_width,
		dst_data, dst_height, dst_width,
		scale_h,scale_w);
	CUDA_POST_KERNEL_CHECK;
  }


  template <typename Dtype>
  __global__ void kernel_ResizeBackward(const int nthreads, const Dtype* top_diff, const int top_step,
                                        Dtype* bottom_diff, const int bottom_step,
                                        const Dtype* loc1, const  Dtype* weight1, const Dtype* loc2, const Dtype* weight2,
                                        const Dtype* loc3, const Dtype* weight3, const Dtype* loc4, const Dtype* weight4) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int bottom_diff_offset = bottom_step*index;
      int top_diff_offset = top_step*index;
      for (int idx = 0; idx < top_step; ++idx) {
        bottom_diff[bottom_diff_offset + int(loc1[idx])] += top_diff[top_diff_offset + idx] * weight1[idx];
        bottom_diff[bottom_diff_offset + int(loc2[idx])] += top_diff[top_diff_offset + idx] * weight2[idx];
        bottom_diff[bottom_diff_offset + int(loc3[idx])] += top_diff[top_diff_offset + idx] * weight3[idx];
        bottom_diff[bottom_diff_offset + int(loc4[idx])] += top_diff[top_diff_offset + idx] * weight4[idx];
      }
    }
  }
  
template <typename Dtype>
__global__ void kernel_GetBiLinearResizeMatRules(const int nthreads,  const int src_height, const int src_width,
		const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
				Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int dst_h = index /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = index %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
//		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = src_idx;
		weight1[dst_idx] = w_h0 * w_w0;

		loc2[dst_idx] = 0;
		weight2[dst_idx] = 0;

		weight3[dst_idx] = 0;
		loc3[dst_idx] = 0;

		loc4[dst_idx] = 0;
		weight4[dst_idx] = 0;

/*		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = src_idx + 1;
			weight2[dst_idx] = w_h0 * w_w1;
//			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}
		if (src_h + 1 < src_height)
		{
//			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = src_idx + src_width;
		}
		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = src_idx + src_width + 1;
			weight4[dst_idx] = w_h1 * w_w1;
//			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}*/

		if (src_w < src_width)
		{
			loc2[dst_idx] = src_idx;
			weight2[dst_idx] = w_h0 * w_w1;
		}
		if (src_h < src_height)
		{
			loc3[dst_idx] = src_idx;
			weight3[dst_idx] = w_h1 * w_w0;	
		}
		if (src_w < src_width && src_h < src_height)
		{
			loc4[dst_idx] = src_idx;
			weight4[dst_idx] = w_h1 * w_w1;
		}

	}
}

  template <typename Dtype>
  void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* top_diff = top[0]->mutable_gpu_diff();

    const Dtype* loc1 = this->locs_[0]->gpu_data();
    const Dtype* weight1 = this->locs_[0]->gpu_diff();
    const Dtype* loc2 = this->locs_[1]->gpu_data();
    const Dtype* weight2 = this->locs_[1]->gpu_diff();
    const Dtype* loc3 = this->locs_[2]->gpu_data();
    const Dtype* weight3 = this->locs_[2]->gpu_diff();
    const Dtype* loc4 = this->locs_[3]->gpu_data();
    const Dtype* weight4 = this->locs_[3]->gpu_diff();

    caffe::caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

	const Dtype scale_w = bottom[0]->width() / (Dtype)top[0]->width();
	const Dtype scale_h = bottom[0]->height() / (Dtype)top[0]->height();
	int loop_n_t = top[0]->height() * top[0]->width();
	kernel_GetBiLinearResizeMatRules<Dtype> << <CAFFE_GET_BLOCKS(loop_n_t), CAFFE_CUDA_NUM_THREADS >> >(
		loop_n_t, bottom[0]->height(), bottom[0]->width(),
		top[0]->height(), top[0]->width(), scale_h, scale_w,
		this->locs_[0]->mutable_gpu_data(), this->locs_[0]->mutable_gpu_diff(),
		this->locs_[1]->mutable_gpu_data(), this->locs_[1]->mutable_gpu_diff(),
		this->locs_[2]->mutable_gpu_data(), this->locs_[2]->mutable_gpu_diff(),
		this->locs_[3]->mutable_gpu_data(), this->locs_[3]->mutable_gpu_diff());

    const int top_step = top[0]->offset(0, 1);
    const int bottom_step = bottom[0]->offset(0, 1);
    int loop_n = this->out_num_ * this->out_channels_;
	kernel_ResizeBackward<Dtype> << <CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >> >(
      loop_n, top_diff, top_step,
      bottom_diff, bottom_step,
      loc1, weight1, loc2, weight2,
      loc3, weight3, loc4, weight4);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}  // namespace caffe