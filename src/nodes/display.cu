#include "nodes/display.h"
#include "core/common_cu.h"

__global__
void GrayPictureGeneratorKernel(const int num_images, const float *in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char *out)
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

__global__
void ColorPictureGeneratorKernel(const int num_images, const float *in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char *out)
{
	int num_image = blockIdx.x*blockDim.x + threadIdx.x;
	if (num_image < num_images) {
		int input_width = per_image_height * per_image_width;

		float max[3];
		float min[3];
		for (int i = 0; i < 3; i++) {
			max[i] = -FLT_MAX;
			min[i] = FLT_MAX;
		}

		float tmp[3];
		for (int i = 0; i < input_width; i++) {
			for (int j = 0; j < 3; j++) {
				tmp[j] = in[(3 * num_image + j)*input_width + i];
				if (max[j] < tmp[j]) max[j] = tmp[j];
				if (min[j] > tmp[j]) min[j] = tmp[j];
			}
		}

		float denom[3];
		for (int i = 0; i < 3; i++) {
			denom[i] = (max[i] - min[i]);
			if (denom[i] == 0)
				denom[i] = 1;			
		}			

		int output_block_col = num_image % num_image_per_row_and_col;
		int output_block_row = (num_image - output_block_col) / num_image_per_row_and_col;
		int output_width = per_image_width * num_image_per_row_and_col * 3;

		for (int i = 0; i < input_width; i++) {
			int input_image_col = i % per_image_width;
			for (int j = 0; j < 3; j++) {				
				int output_image_row = (i - input_image_col) / per_image_width;
				int input_image_row = output_image_row * 3 + j;
				int output_col = 3 * (output_block_col*per_image_width + input_image_col) + j;
				int output_row = output_block_row*per_image_height + output_image_row;
				out[output_row*output_width + output_col] = (in[(3 * num_image + 2 - j)*input_width + i] - min[2 - j]) / denom[2 - j] * 255;
			}
		}

	}
}

Display::Display(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_display_param() == false) << "param.has_display_param() == false";
}

void Display::initForward() {
	auto display_param = _param.display_param();
	_delay_msec = display_param.delay_msec();
	_display_type = display_param.display_type();
	_display_time = display_param.display_time();
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
	LOG(INFO) << "Initializing Display " << _name << " - " << _inputs[0]->value()->shape() << " -> " << _outputs[0]->value()->shape() << "(" << (num_channels == 3? "COLOR": "GRAY") <<  ")";

	disp = cv::Mat(pic_height, pic_width, (num_channels == 3 ? CV_8UC3 : CV_8U));	
}

void Display::initBackward() {
	
}

void Display::forward() {
	if (_display_time == deepflow::ActionTime::END_OF_EPOCH && _context->last_batch == false)
		return;
	if (_display_type == deepflow::ActionType::VALUES) {
		if (num_channels == 3) {
			ColorPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images, (float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col,(unsigned char*) _outputs[0]->value()->mutableData());
		}
		else
			GrayPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images,(float*)_inputs[0]->value()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
		DF_KERNEL_CHECK();				
		DF_NODE_CUDA_CHECK(cudaMemcpy(disp.ptr<uchar>(), _outputs[0]->value()->data(), num_pic_pixels, cudaMemcpyDeviceToHost));
		cv::imshow(name(), disp);		
		int key = cv::waitKey(_delay_msec);
		if (key == 27) {
			if (_context)
				_context->quit = true;
			else
				exit(0);
		}
	}
	if (_display_type == deepflow::ActionType::DIFFS) {
		if (num_channels == 3)
			ColorPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images, (float*)_inputs[0]->diff()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
		else
			GrayPictureGeneratorKernel << < numOfBlocks(num_images), maxThreadsPerBlock >> >(num_images,(float*)_inputs[0]->diff()->data(), per_image_height, per_image_width, num_image_per_row_and_col, (unsigned char*)_outputs[0]->value()->mutableData());
		DF_KERNEL_CHECK();
		DF_NODE_CUDA_CHECK(cudaMemcpy(disp.ptr<uchar>(), _outputs[0]->value()->data(), num_pic_pixels, cudaMemcpyDeviceToHost));
		cv::imshow(name(), disp);
		int key = cv::waitKey(_delay_msec);
		if (key == 27) {
			if (_context)
				_context->quit = true;
			else
				exit(0);
		}
	}
}

void Display::backward() {
	LOG(FATAL);
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
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
