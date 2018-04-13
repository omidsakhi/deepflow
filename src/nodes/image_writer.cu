#include "nodes/image_writer.h"
#include "core/common_cu.h"

#include <opencv2/opencv.hpp>

ImageWriter::ImageWriter(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_image_writer_param() == false) << "param.has_image_writer_param() == false";
}

void ImageWriter::init() {

	auto dims = _inputs[0]->value()->dims();
	input_size = _inputs[0]->value()->size();
	input_size_in_bytes = _inputs[0]->value()->sizeInBytes();
	num_channels = dims[1];
	num_samples = dims[0];
	num_images = num_samples * (num_channels == 3 ? 1 : num_channels);
	per_image_height = dims[2];
	per_image_width = dims[3];
	num_image_per_row_and_col = (int)floor(sqrt((float)num_images));
	pic_width = per_image_width * num_image_per_row_and_col;
	pic_height = per_image_height * ((int)ceil(((float)num_images / num_image_per_row_and_col)));
	num_pic_pixels = pic_width * pic_height * (num_channels == 3 ? 3 : 1);	
	_filename = _param->image_writer_param().filename();
	_outputs[0]->initValue({ 1 , (num_channels == 3 ? 3 : 1), pic_height, pic_width }, Tensor::Int8);
	disp = std::make_shared<cv::Mat>(cv::Mat(pic_height, pic_width, (num_channels == 3 ? CV_8UC3 : CV_8U)));
}

void ImageWriter::forward() {
	if (num_channels == 3)
		ColorPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images, (float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
	else
		GrayPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images, (float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
	DF_NODE_CUDA_CHECK(cudaMemcpy(disp->ptr<uchar>(), _outputs[0]->value()->data(), num_pic_pixels, cudaMemcpyDeviceToHost));	
	auto filename = _filename;
	auto start_pos = filename.find("{it}");
	if (start_pos != std::string::npos) {
		filename.replace(start_pos, 4, std::to_string(_context->current_iteration));
	}
	if (_context && !_text_line.empty()) {
		cv::putText(*disp.get(), _text_line, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8, false);
	}
	cv::imwrite(filename + ".jpg", *disp.get());
}

void ImageWriter::backward() {
	
}

std::string ImageWriter::to_cpp() const
{
	return "";
}

void ImageWriter::set_text_line(std::string text)
{
	_text_line = text;
}

void ImageWriter::set_filename(std::string filename)
{
	_filename = filename;
}
