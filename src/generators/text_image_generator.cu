#include "generators/text_image_generator.h"
#include "nodes/variable.h"
#include "core/initializer.h"

#include <opencv2/opencv.hpp>

__global__
void TextImageGeneratorKernel(const int n, const int offset, const unsigned char *text_image, const int height, const int width, float *output)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int channel = i % 3;
		int input_pixel = (i - channel) / 3;
		int input_col = input_pixel % width;
		int input_row = (input_pixel - input_col) / width;
		int index = (2 - channel) * width * height + input_row * width + input_col + offset;
		float old_value = output[index];
		float new_value = (((float)text_image[i] / 255.0f) - 0.5f) * 2;
		output[index] = (new_value > old_value) ? new_value : old_value;

	}
}


TextImageGenerator::TextImageGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam * param) : Variable(initializer, param) {
	LOG_IF(FATAL, param->has_text_image_generator_param() == false) << "param->has_text_image_generator_param() == false";	
}

bool TextImageGenerator::is_generator()
{
	return true;
}

void TextImageGenerator::init()
{
	_n = _param->variable_param().init_param().tensor_param().dims(0);
	_c = _param->variable_param().init_param().tensor_param().dims(1);
	_h = _param->variable_param().init_param().tensor_param().dims(2);
	_w = _param->variable_param().init_param().tensor_param().dims(3);
	LOG_IF(FATAL, _n < 1) << "Text Image Generator batch size must be more than 0.";
	LOG_IF(FATAL, _c != 3) << "Text Image Generator channel size must be 3.";
	LOG_IF(FATAL, !_param->variable_param().solver_name().empty()) << "Text Image Generator does not support a solver.";
	_dict = { "AaBbCcDdEeFfGgHhIiJjKkLlMmOoPpQqRrSsTtUuVvWwXxYyZz0123456789-_/\.%" };
	Variable::init();
	cv_img = std::make_shared<cv::Mat>(cv::Mat(_h, _w, CV_8UC3));
	DF_NODE_CUDA_CHECK(cudaMalloc(&d_img, _h * _w * 3));
}

void TextImageGenerator::forward()
{
	_initializer->apply(this);
	for (int index = 0; index < _n; ++index)
	{
		*cv_img.get() = 0;
		std::string text;
		int total_draw = rand() % 10;
		for (int c = 0; c < total_draw; c++) {
			text += _dict.at(rand() % _dict.size());
		}		
		auto org = cv::Point(rand() % (_w/2), rand() % (_h - 30) + 15);
		int fontFace = rand() % 8;
		double fontScale = (float)rand() / RAND_MAX * 2 + 1;
		auto color = cv::Scalar(255, 255, 255);
		auto thickness = rand() % 2 + 1;
		cv::putText(*cv_img.get(), text, org, fontFace, fontScale, color, thickness);
		auto size = _h * _w * 3;
		DF_NODE_CUDA_CHECK(cudaMemcpy(d_img, cv_img->ptr<uchar>(), size, cudaMemcpyHostToDevice));
		TextImageGeneratorKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, index * size, d_img, _h, _w, (float*) _outputs[0]->value()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string TextImageGenerator::to_cpp() const
{
	return std::string();
}
