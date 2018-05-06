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


TextImageGenerator::TextImageGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam * param) : Node(param) {
	LOG_IF(FATAL, param->has_text_image_generator_param() == false) << "param->has_text_image_generator_param() == false";	
	_initializer = initializer;
}

bool TextImageGenerator::is_generator()
{
	return true;
}

void TextImageGenerator::init()
{
	auto variabl_param = _param->variable_param();
	auto tig_param = _param->text_image_generator_param();	
	_n = variabl_param.init_param().tensor_param().dims(0);
	_c = variabl_param.init_param().tensor_param().dims(1);
	_h = variabl_param.init_param().tensor_param().dims(2);
	_w = variabl_param.init_param().tensor_param().dims(3);
	LOG_IF(FATAL, _n < 1) << "Text Image Generator batch size must be more than 0.";
	LOG_IF(FATAL, _c != 3) << "Text Image Generator channel size must be 3.";
	LOG_IF(FATAL, !variabl_param.solver_name().empty()) << "Text Image Generator does not support a solver.";
	_chars = tig_param.chars();	
	_initializer->init();
	_outputs[0]->initValue(_initializer->dims());
	_outputs[1]->initValue(_initializer->dims());	
	cv_img_actual = std::make_shared<cv::Mat>(cv::Mat(_h, _w, CV_8UC3));
	cv_img_target = std::make_shared<cv::Mat>(cv::Mat(_h, _w, CV_8UC3));
	DF_NODE_CUDA_CHECK(cudaMalloc(&d_img_actual, _h * _w * 3));
	DF_NODE_CUDA_CHECK(cudaMalloc(&d_img_target, _h * _w * 3));
}

void TextImageGenerator::forward()
{
	_initializer->apply(this);
	for (int index = 0; index < _n; ++index)
	{
		*cv_img_actual.get() = 0;
		*cv_img_target.get() = 0;
		int text_lines = rand() % 5 + 1;
		int y = rand() % (_h / 2) + 1;
		int x = rand() % (_w / 2) + 1;
		for (int i = 0; i < text_lines; ++i) {			
			int fontFace = rand() % 2;
			double fontScale = (float)rand() / RAND_MAX * 2.5 + 1.0;
			auto randColor = rand() % 128 + 128; 
			auto color = cv::Scalar(randColor, randColor, randColor);
			auto thickness = rand() % 3 + 1;
			std::string text;
			if (!_chars.empty())
				text = generate_random_text(10, _chars);
			else {
				auto tig = _param->text_image_generator_param();
				text = tig.words(rand() % tig.words_size());
			}
			int baseLine = 0;
			auto textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
			y += textSize.height +5 + rand() % 20;
			auto org = cv::Point(x, y);			
			cv::putText(*cv_img_actual.get(), text, org, fontFace, fontScale, color, thickness);
			cv::putText(*cv_img_target.get(), text, org, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
			
			auto drawUnderline = rand() % 2;
			if (drawUnderline)
			{
				auto randColor = rand() % 128 + 128;
				auto color = cv::Scalar(randColor, randColor, randColor);
				cv::Point startPoint = org;
				cv::Point endPoint = org;
				auto lineThickness = rand() % 3 + 2;
				y += rand() % 5 + lineThickness;
				startPoint.y = y;
				endPoint.y = y;
				endPoint.x += textSize.width;				
				cv::line(*cv_img_actual.get(), startPoint, endPoint, color, lineThickness);
			}
			
		}
		auto size = _h * _w * 3;
		DF_NODE_CUDA_CHECK(cudaMemcpy(d_img_actual, cv_img_actual->ptr<uchar>(), size, cudaMemcpyHostToDevice));
		DF_NODE_CUDA_CHECK(cudaMemcpy(d_img_target, cv_img_target->ptr<uchar>(), size, cudaMemcpyHostToDevice));
		TextImageGeneratorKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, index * size, d_img_actual, _h, _w, (float*) _outputs[0]->value()->gpu_data(DF_LINE));
		DF_KERNEL_CHECK();
		TextImageGeneratorKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, index * size, d_img_target, _h, _w, (float*)_outputs[1]->value()->gpu_data(DF_LINE));
		DF_KERNEL_CHECK();
	}
}

std::string TextImageGenerator::to_cpp() const
{
	return std::string();
}

std::string TextImageGenerator::generate_random_text(int max_characters, std::string &dict)
{
	std::string text;
	int total_draw = rand() % max_characters + 1;
	for (int c = 0; c < total_draw; c++) {
		text += dict.at(rand() % dict.size());
	}
	return text;
}