#include "nodes/gaussian_kernel.h"

__global__
void GaussianWeightsKernel(const int n, const float sigma, const int window_size, const int num_channels, float * w)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int indexTemp = i;
		const int x = indexTemp % window_size;
		indexTemp /= window_size;
		const int y = indexTemp % window_size;
		indexTemp /= window_size;
		const int input_channel = indexTemp % num_channels;
		indexTemp /= num_channels;
		const int output_channel = indexTemp;
		if (output_channel == input_channel) {
			float half_window_size = (float)window_size / 2.0f;
			if (sigma != 0) {				
				float xx = (x - half_window_size) * (x - half_window_size);
				float yy = (y - half_window_size) * (y - half_window_size);
				float ss = sigma * sigma;
				w[i] = 1.0f / (2.0f * 3.141592f * ss) * exp(-0.5f * (xx + yy) / ss);
			}
		}
		else {
			w[i] = 0;
		}
	}
}

ConvGaussianKernel::ConvGaussianKernel(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_gaussian_kernel_param() == false) << "param.has_gaussian_kernel_param() == false";
}

void ConvGaussianKernel::setSigma(float value)
{
	if (value != _current_sigma) {
		_current_sigma = value;
		_param->mutable_gaussian_kernel_param()->set_sigma(_current_sigma);
		generate();
	}
}

void ConvGaussianKernel::init()
{
	auto gparam = _param->gaussian_kernel_param();
	_window_size = gparam.window_size();
	_current_sigma = gparam.sigma();
	if (gparam.num_channels())
		_num_channels = gparam.num_channels();	
	std::array<int, 4> dims = { _num_channels, _num_channels, _window_size, _window_size};
	_outputs[0]->initValue(dims);
	generate();
}

void ConvGaussianKernel::forward()
{
}

void ConvGaussianKernel::backward()
{
}

std::string ConvGaussianKernel::to_cpp() const
{
	return std::string();
}

void ConvGaussianKernel::generate()
{
	auto size = _outputs[0]->value()->size();
	GaussianWeightsKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _current_sigma, _window_size, _num_channels, (float*)_outputs[0]->value()->gpu_data(DF_LINE));
	DF_KERNEL_CHECK();
}
