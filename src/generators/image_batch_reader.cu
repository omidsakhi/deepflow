#include "generators/image_batch_reader.h"
#include "core/common_cu.h"
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>

__global__
void GrayImageBatchReaderKernel(const int n, const int offset, const bool between_0_and_1, const unsigned char *in, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		out[i + offset] = between_0_and_1 ? (float)in[i] / 255.0f : (((float)in[i] / 255.0f) - 0.5f) * 2;
	}
}

__global__
void ColorImageBatchReaderKernel(const int n, const int offset, const bool between_0_and_1, const unsigned char *in, const int width, const int height, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int channel = i % 3;
		int input_pixel = (i - channel) / 3;
		int input_col = input_pixel % width;
		int input_row = (input_pixel - input_col) / width;
		out[(2 - channel) * width * height + input_row * width + input_col + offset] = between_0_and_1 ? (float)in[i] / 255.0f: (((float)in[i] / 255.0f) - 0.5f) * 2;
	}
}


ImageBatchReader::ImageBatchReader(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_image_batch_reader_param() == false) << "param.has_image_batch_reader_param() == false";
}

void ImageBatchReader::init() {	
	auto image_batch_reader_param = _param->image_batch_reader_param();
	_folder_path = image_batch_reader_param.folder_path();		
	std::experimental::filesystem::path path(_folder_path);	
	std::experimental::filesystem::directory_iterator dir(path);
	for (auto &p : dir) {
		auto path = p.path();
		std::string ext = path.extension().string();
		if (ext == ".png" || ext == ".jpg" || ext == ".pgm")
			_list_of_files.push_back(path);
	}
	LOG_IF(FATAL, _list_of_files.size() == 0) << "Failed to find .png or .jpg or .pgm image files in the folder " << _folder_path;
	_num_total_samples = _list_of_files.size();
	_randomize = image_batch_reader_param.randomize();
	_between_0_and_1 = image_batch_reader_param.between_0_and_1();
	const deepflow::TensorParam &tensorParam = image_batch_reader_param.tensor_param();
	switch (tensorParam.dims_size()) {
	case 1:
		_dims = { 1,tensorParam.dims(0),1,1 };
		break;
	case 2:
		_dims = { tensorParam.dims(0),tensorParam.dims(1),1,1 };
		break;
	case 3:
		_dims = { tensorParam.dims(0), 1, tensorParam.dims(1),tensorParam.dims(2) };
		break;
	case 4:
		_dims = { tensorParam.dims(0),tensorParam.dims(1),tensorParam.dims(2),tensorParam.dims(3) };
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}
	_batch_size = _dims[0];	
	_num_batches = _num_total_samples / _batch_size;
	_last_batch = (_current_batch == (_num_batches - 1));
	_outputs[0]->initValue(_dims);	
	_indices.resize(_batch_size);	
}

void thread_internal_read_image(int img_index, std::string file_name, float * output, int channels, int height, int width, int num_of_blocks, int max_thread_per_block, bool between_0_and_1)
{
	cv::Mat img;
	if (channels == 1)
		img = cv::imread(file_name, 0);
	else if (channels == 3)
		img = cv::imread(file_name);
	else
		LOG(FATAL) << "Unsupported channel size.";
	LOG_IF(FATAL, img.empty()) << "Image " << file_name << " does not exist.";	
	LOG_IF(FATAL, img.channels() != channels) << "Provided channels doesn't match for " << file_name;
	LOG_IF(FATAL, img.rows != height) << "Provided height doesn't match for " << file_name;
	LOG_IF(FATAL, img.cols != width) << "Provided width doesn't match for " << file_name;
	size_t img_size = img.cols * img.rows * img.channels();
	unsigned char *d_img;
	cudaMalloc(&d_img, img_size);
	DF_CUDA_CHECK(cudaMemcpy(d_img, img.ptr<uchar>(), img_size, cudaMemcpyHostToDevice));	
	if (channels == 1) {
		GrayImageBatchReaderKernel << < num_of_blocks, max_thread_per_block >> > (img_size, img_index * img_size, between_0_and_1, d_img, output);
		DF_KERNEL_CHECK();
	}
	else if (channels == 3) {
		ColorImageBatchReaderKernel << < num_of_blocks, max_thread_per_block >> > (img_size, img_index * img_size, between_0_and_1, d_img, img.cols, img.rows, output);
		DF_KERNEL_CHECK();
	}	
	img.release();
	cudaFree(d_img);
}

void ImageBatchReader::forward()
{
	_last_batch = (_current_batch >= (_num_batches - 1));
	if (_last_batch) {
		_current_batch = 0;
	}
	else {
		_current_batch++;
	}	

	if (_randomize) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, _num_total_samples - 1);
		for (int i = 0; i < _indices.size(); ++i)
			_indices[i] = dis(gen);
	}
	else {
		for (int i = _current_batch*_dims[0]; i < (_current_batch + 1)*_dims[0]; i++) {
			_indices[i%_dims[0]] = i;
		}
	}

	std::list<std::thread> thread_list;
	for (int index = 0; index < _indices.size(); ++index)
	{
		std::string file_name = _list_of_files[_indices[index]].string();
		thread_list.push_back(std::thread(thread_internal_read_image, index, file_name, (float*)_outputs[0]->value()->mutableData(), _dims[1], _dims[2], _dims[3], numOfBlocks(_dims[1] * _dims[2] * _dims[3]), maxThreadsPerBlock, _between_0_and_1));
	}
	for (auto &thread : thread_list)
		thread.join();
}

bool ImageBatchReader::is_last_batch() {
	return _last_batch;
}

std::string ImageBatchReader::to_cpp() const
{
	auto image_batch_reader_param = _param->image_batch_reader_param();
	auto folder_path = image_batch_reader_param.folder_path();	
	auto type = image_batch_reader_param.tensor_param().type();
	std::string cpp = "auto " + _name + " = df.image_batch_reader(\"" + folder_path + "\", ";
	if (type == deepflow::ImageReaderParam_Type_GRAY_ONLY)
		cpp += "ImageReaderParam_Type_GRAY_ONLY, ";
	else
		cpp += "ImageReaderParam_Type_COLOR_IF_AVAILABLE, ";
	cpp += (_randomize ? "true" : "false") + std::string(", ");
	cpp += "\"" + _name + "\");";	
	return cpp;
}
