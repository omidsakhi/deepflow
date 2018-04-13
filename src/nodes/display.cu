#include "nodes/display.h"
#include "core/common_cu.h"

Display::Display(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_display_param() == false) << "param.has_display_param() == false";
}

void Display::init() {
	auto display_param = _param->display_param();
	_delay_msec = display_param.delay_msec();
	_display_type = display_param.display_type();	
	_draw_iteration = display_param.draw_iteration();
	_epoch_frequency = display_param.epoch_frequency();
	_iter_frequency = display_param.iter_frequency();
	auto dims = _inputs[0]->value()->dims();	
	input_size = _inputs[0]->value()->size();
	input_size_in_bytes = _inputs[0]->value()->sizeInBytes();
	num_channels = dims[1]; 
	num_samples = dims[0];	
	num_images = num_samples * (num_channels == 3? 1 : num_channels);
	per_image_height = dims[2];
	per_image_width = dims[3];
	num_image_per_row_and_col = (int)floor(sqrt((float)num_images));
	pic_width = per_image_width * num_image_per_row_and_col;
	pic_height = per_image_height * ((int)ceil(((float)num_images / num_image_per_row_and_col)));
	num_pic_pixels = pic_width * pic_height * (num_channels == 3 ? 3 : 1);	

	_outputs[0]->initValue({ 1 , (num_channels == 3 ? 3 : 1), pic_height, pic_width }, Tensor::Int8);	

	disp = cv::Mat(pic_height, pic_width, (num_channels == 3 ? CV_8UC3 : CV_8U));	
}

void Display::forward() {
	if (_enabled == false)
		return;
	if (_context) {
		if (_context->current_epoch % _epoch_frequency != 0)
			return;
		if (_context->current_iteration % _iter_frequency != 0)
			return;
	}
	if (_display_type == deepflow::ActionType::VALUES) {
		if (num_channels == 3) {
			ColorPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images, (float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col,(unsigned char*) _outputs[0]->value()->mutableData());
		}
		else
			GrayPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images,(float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
	}
	if (_display_type == deepflow::ActionType::DIFFS) {
		if (num_channels == 3)
			ColorPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images, (float*)_inputs[0]->diff()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
		else
			GrayPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images,(float*)_inputs[0]->diff()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
	}
	DF_KERNEL_CHECK();
	DF_NODE_CUDA_CHECK(cudaMemcpy(disp.ptr<uchar>(), _outputs[0]->value()->data(), num_pic_pixels, cudaMemcpyDeviceToHost));
	if (_context && _draw_iteration) {
		cv::putText(disp, std::string("Iteration: ") + std::to_string(_context->current_iteration), cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8, false);
	}
	cv::imshow(name(), disp);
	int key = cv::waitKey(_delay_msec);
	if (key == 27) {
		if (_context)
			_context->quit = true;
		else
			exit(0);
	}
	else if (key == 110 && _delay_msec > 0) {
		// UP 
		_delay_msec = 1;
	}
	else if (key == 109) {
		_delay_msec+=100;
	}
}

void Display::backward() {
	
}

std::string Display::to_cpp() const
{	
	std::string cpp = "df.display(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(_delay_msec) + ", ";
	if (_display_type == deepflow::ActionType::DIFFS) {
		cpp += "deepflow::DisplayParam_DisplayType_DIFFS, ";
	}
	else {
		cpp += "deepflow::DisplayParam_DisplayType_VALUES, ";
	}	
	cpp += "\"" + _name + "\");";	
	return cpp;
}

void Display::setEnabled(bool state)
{
	_enabled = state;
}
