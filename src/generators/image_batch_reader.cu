#include "generators/image_batch_reader.h"
#include "core/common_cu.h"
#include <opencv2/opencv.hpp>

__global__
void ImageBatchReaderNormalizeKernel(const int n, const unsigned char *in, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = (((float)in[i] / 255.0f) - 0.5f) * 2;
	}
}

__global__
void ImageBatchReaderConvertOpenCV3ImageKernel(const int n, const unsigned char *in, const int width, const int height, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int ch = i % 3;
		int i3 = (i - ch) / 3;
		int i3c = i3 % width;
		int i3r = (i3 - i3c) / height;
		out[(2 - ch) * width * height + i3r * width + i3c] = (((float)in[i] / 255.0f) - 0.5f) * 2;
	}
}


ImageBatchReader::ImageBatchReader(const deepflow::NodeParam &param) : Node(param), Generator(param) {
	LOG_IF(FATAL, param.generator_param().has_image_batch_reader_param() == false) << "param.generator_param().has_image_batch_reader_param() == false";
}

void ImageBatchReader::nextBatch() {
		_current_batch = (_last_batch == true) ? 0 : _current_batch + 1;
		size_t _remaining_samples = _num_total_samples - _current_batch*_dims[0];
		if (_remaining_samples > _dims[0])
		{
			_num_batch_samples = _dims[0];
			_last_batch = false;
		}
		else
		{
			_num_batch_samples = _remaining_samples;
			_last_batch = true;
		}
		for (int i = _current_batch*_dims[0]; i < (_current_batch + 1)*_dims[0]; i++) {
			std::string file_name = _list_of_files[i].string();
			if (_dims[1] == 1)
				img = cv::imread(file_name, 0);
			else if (_dims[1] == 3)
				img = cv::imread(file_name);
			else
				LOG(FATAL) << "Unsupported channel size.";
			LOG_IF(FATAL, img.empty()) << "Image " << file_name << " does not exist.";
			LOG_IF(FATAL, img.channels() != _dims[1]) << "Provided channels doesn't match for " << file_name;
			LOG_IF(FATAL, img.rows != _dims[2]) << "Provided height doesn't match for " << file_name;
			LOG_IF(FATAL, img.cols != _dims[3]) << "Provided width doesn't match for " << file_name;
			size_t size = _outputs[0]->value()->size();
			DF_CUDA_CHECK(cudaMemcpy(d_img, img.ptr<uchar>(), size, cudaMemcpyHostToDevice));
			if (img.channels() == 1) {
				ImageBatchReaderNormalizeKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, d_img, (float*)_outputs[0]->value()->mutableData());
				DF_KERNEL_CHECK();
			}
			else if (img.channels() == 3) {
				ImageBatchReaderConvertOpenCV3ImageKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, d_img, img.cols, img.rows, (float*)_outputs[0]->value()->mutableData());
				DF_KERNEL_CHECK();
			}
		}
}

void ImageBatchReader::initForward() {
	auto image_batch_reader_param = _param.generator_param().image_batch_reader_param();
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
	_last_batch = false;
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
	LOG_IF(FATAL, _dims[0] != 1) << "Batch size is limited to 1.";
	_outputs[0]->initValue(_dims);
	LOG(INFO) << "Initializing image_batch_reader from folder " << _folder_path << " - " << _outputs[0]->value()->shape();
	size_t size = _outputs[0]->value()->size();
	DF_CUDA_CHECK(cudaMalloc(&d_img, size));
}

bool ImageBatchReader::isLastBatch() {
	return _last_batch;
}

std::string ImageBatchReader::to_cpp() const
{
	auto image_reader_param = _param.generator_param().image_reader_param();
	auto file_name = image_reader_param.file_name();
	auto type = image_reader_param.type();
	std::string cpp = "auto " + _name + " = df.image_batch_reader(\"" + file_name + "\", ";
	if (type == deepflow::ImageReaderParam_Type_GRAY_ONLY)
		cpp += "ImageReaderParam_Type_GRAY_ONLY, ";
	else
		cpp += "ImageReaderParam_Type_COLOR_IF_AVAILABLE, ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
