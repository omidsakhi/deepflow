#include "generators/image_reader.h"
#include "core/common_cu.h"
#include <opencv2/opencv.hpp>

__global__
void ImageReaderKernel(const int n, const unsigned char *in, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = (((float)in[i] / 255.0f) - 0.5f) * 2;
	}
}


ImageReader::ImageReader(const NodeParam &param) : Node(param), Generator(param) {
	LOG_IF(FATAL, param.generator_param().has_image_reader_param() == false) << "param.generator_param().has_image_reader_param() == false";
}

void ImageReader::nextBatch() {

}

void ImageReader::initForward() {
	
	auto file_name = _param.generator_param().image_reader_param().file_name();
	img = cv::imread(file_name, 0);
	LOG_IF(FATAL, img.empty()) << "Image " << file_name << "does not exist.";		
	_outputs[0]->initValue({ 1, img.channels(), img.rows , img.cols });
	LOG(INFO) << "Initializing Image " << _name << " - " << _outputs[0]->value()->shape();
	size_t size = _outputs[0]->value()->size();
	//cudaStream_t stream;
	//DF_CUDA_CHECK(cudaStreamCreate(&stream));
 	ImageReaderKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, img.ptr<uchar>(), (float*) _outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
	//DF_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ImageReader::initBackward() {
	_outputs[0]->initDiff();
}

void ImageReader::forward() {

}

void ImageReader::backward() {

}

bool ImageReader::isLastBatch() {
	return _last_batch;
}
