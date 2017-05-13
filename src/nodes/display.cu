#include "nodes/display.h"
#include "core/common_cu.h"

#include <opencv2/opencv.hpp>

__global__
void PictureGeneratorKernel(const int num_images, const float *in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char *out)
{
	int num_image = blockIdx.x*blockDim.x + threadIdx.x;
	if (num_image < num_images) {
		int input_width = per_image_height * per_image_width;

		float max = -FLT_MAX;
		float min = FLT_MIN;

		float tmp;
		for (int i = 0; i < input_width; i++) {
			tmp = in[num_image*input_width + i];
			if (max < tmp)
				max = tmp;
			if (min > tmp)
				min = tmp;
		}

		float denom = (max - min);
		if (denom == 0)
			denom = 1;

		int output_block_col = num_image % num_image_per_row_and_col;
		int output_block_row = (num_image - output_block_col) / num_image_per_row_and_col;
		int output_width = per_image_width * num_image_per_row_and_col;

		for (int i = 0; i < input_width; i++) {
			int input_image_col = i % per_image_width;
			int input_image_row = (i - input_image_col) / per_image_width;
			int output_col = output_block_col*per_image_width + input_image_col;
			int output_row = output_block_row*per_image_height + input_image_row;
			out[output_row*output_width + output_col] = (in[num_image*input_width + i] - min) / denom * 255;
		}
	}
}

Display::Display(const deepflow::NodeParam &_block_param) : Node(_block_param) {
	LOG_IF(FATAL, _block_param.has_display_param() == false) << "param.has_display_param() == false";
}

void Display::initForward() {
	_delay_msec = _param.display_param().delay_msec();
	_display_type = _param.display_param().display_type();
	auto dims = _inputs[0]->value()->dims();	
	input_size = _inputs[0]->value()->size();
	input_size_in_bytes = _inputs[0]->value()->sizeInBytes();
	num_channels = dims[1];
	num_samples = dims[0];
	num_images = num_samples * num_channels;
	per_image_height = dims[2];
	per_image_width = dims[3];
	num_image_per_row_and_col = (int)floor(sqrt((float)num_images));
	pic_width = per_image_width * num_image_per_row_and_col;
	pic_height = per_image_height * ((int)ceil(((float)num_images / num_image_per_row_and_col)));
	num_pic_pixels = pic_width * pic_height;	

	LOG_IF(FATAL, cudaMalloc(&d_pic, sizeof(unsigned char) * num_pic_pixels) != 0);
	LOG_IF(FATAL, cudaMemset(d_pic, 0, sizeof(unsigned char) * num_pic_pixels) != 0);
			
	LOG(INFO) << "Initializing Display " << _name << " - " << _inputs[0]->value()->shape() << " -> " << pic_height << "x" << pic_width;
	disp = cv::Mat(pic_height, pic_width, CV_8U);	
}

void Display::initBackward() {
	
}

void Display::forward() {
	if (_display_type == deepflow::DisplayParam_DisplayType_VALUES) {
		PictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images,(float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col, d_pic);
		DF_KERNEL_CHECK();
		cudaDeviceSynchronize();
		DF_CUDA_CHECK(cudaMemcpy(disp.ptr<uchar>(), d_pic, sizeof(unsigned char) * num_pic_pixels, cudaMemcpyDeviceToHost));
		cv::imshow(name(), disp);		
		int key = cv::waitKey(_delay_msec);
		if (key == 27) {
			_context->quit = true;
		}
	}
	if (_display_type == deepflow::DisplayParam_DisplayType_DIFFS) {
		PictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images,(float*)_inputs[0]->diff()->data(), per_image_height, per_image_width, num_image_per_row_and_col, d_pic);
		DF_KERNEL_CHECK();
		DF_CUDA_CHECK(cudaMemcpy(disp.ptr<uchar>(), d_pic, sizeof(unsigned char) * num_pic_pixels, cudaMemcpyDeviceToHost));
		cv::imshow(name(), disp);
		int key = cv::waitKey(_delay_msec);
		if (key == 27) {
			_context->quit = true;
		}
	}
}

void Display::backward() {
}

std::string Display::to_cpp() const
{	
	std::string cpp = "df.display(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(_delay_msec) + ", ";
	if (_display_type == deepflow::DisplayParam_DisplayType_DIFFS) {
		cpp += "deepflow::DisplayParam_DisplayType_DIFFS, ";
	}
	else {
		cpp += "deepflow::DisplayParam_DisplayType_VALUES, ";
	}	
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
