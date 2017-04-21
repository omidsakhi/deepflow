#include "ops/display.h"
#include "core/common_cu.h"

#include <opencv2/opencv.hpp>

__global__
void PictureGeneratorKernel(const int n, const float *in, unsigned char *out, const int picWidth, const int sq, const int numImages, const int perImageWidth, const int perImageHeight)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int flat_pixel = i % (perImageWidth * perImageHeight);
		int num_image = (i - flat_pixel) / (perImageWidth * perImageHeight);
		int csmall = flat_pixel % perImageWidth;
		int rsmall = (flat_pixel - csmall) / perImageWidth;
		int cbig = num_image % sq;
		int rbig = (num_image - cbig) / sq;
		int c = cbig*perImageWidth + csmall;
		int r = rbig*perImageHeight + rsmall;
		out[r*picWidth + c] = (in[i] + 1.0f) / 2.0f * 255;
	}
}

Display::Display(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_display_param() == false) << "param.has_display_param() == false";
}

void Display::initForward() {
	_delay_msec = _param.display_param().delay_msec();
	auto dims = _inputs[0]->value()->dims();	
	input_size = _inputs[0]->value()->size();
	input_size_in_bytes = _inputs[0]->value()->sizeInBytes();
	num_images = dims[0];
	per_image_height = dims[2];
	per_image_width = dims[3];
	num_image_per_row_and_col = (int)floor(sqrt((float)num_images));
	pic_width = per_image_width * num_image_per_row_and_col;
	pic_height = per_image_height * ((int)ceil(((float)num_images / num_image_per_row_and_col)));
	num_pic_pixels = pic_width * pic_height;	

	LOG_IF(FATAL, cudaMalloc(&d_pic, sizeof(unsigned char) * num_pic_pixels) != 0);
	LOG_IF(FATAL, cudaMemset(d_pic, 0, sizeof(unsigned char) * num_pic_pixels) != 0);
			
	LOG(INFO) << "Initializing Display " << _name << " - " << _inputs[0]->value()->shape() << " -> " << pic_width << "x" << pic_height;
	disp = cv::Mat(pic_height, pic_width, CV_8U);	
}

void Display::initBackward() {
	
}

void Display::forward() {	
	PictureGeneratorKernel << < numOfBlocks(input_size), maxThreadsPerBlock >> >(input_size, (float*) _inputs[0]->value()->data(), d_pic, pic_width, num_image_per_row_and_col, num_images, per_image_width, per_image_height);
	DF_KERNEL_CHECK();
	DF_CUDA_CHECK(cudaMemcpy(disp.ptr<uchar>(), d_pic, sizeof(unsigned char) *num_pic_pixels, cudaMemcpyDeviceToHost));	
	cv::imshow(name(), disp);
	int key = cv::waitKey(_delay_msec);
	if (key == 27) {
		_context->quit = true;
	}
}

void Display::backward() {
	
}