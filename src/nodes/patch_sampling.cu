#include "nodes/patch_sampling.h"

#include <random>

__global__
void PatchSamplingForward(const int out_size, const int num_channels, const int in_height, const int in_width, const int out_height, const int out_width, const int *sampled_x, const int *sampled_y, const float *in, float * out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < out_size) {
		int indexTemp = i;
		const int ox = indexTemp % out_width;
		indexTemp /= out_width;
		const int oy = indexTemp % out_height;
		indexTemp /= out_height;
		const int c = indexTemp % num_channels;
		indexTemp /= num_channels;
		const int n = indexTemp;
		const int ix = sampled_x[n] + ox;
		const int iy = sampled_y[n] + oy;		
		const int input_index = n * (num_channels * in_height * in_width) + c * (in_height * in_width) + iy * in_width + ix;		
		out[i] = in[input_index];
	}
}

__global__
void PatchSamplingBackward(const int out_size, const int num_channels, const int in_height, const int in_width, const int out_height, const int out_width, const int *sampled_x, const int *sampled_y, const float *dy, float *dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < out_size) {
		int indexTemp = i;
		const int ox = indexTemp % out_width;
		indexTemp /= out_width;
		const int oy = indexTemp % out_height;
		indexTemp /= out_height;
		const int c = indexTemp % num_channels;
		indexTemp /= num_channels;
		const int n = indexTemp;
		const int ix = sampled_x[n] + ox;
		const int iy = sampled_y[n] + oy;
		const int input_index = n * (num_channels * in_height * in_width) + c * (in_height * in_width) + iy * in_width + ix;
		atomicAdd(&dx[input_index], dy[i]);
	}
}

PatchSampling::PatchSampling(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_patch_sampling_param() == false) << "param.has_patch_sampling_param() == false";
}

void PatchSampling::init()
{
	auto param = _param->patch_sampling_param();	
	auto inputDim = _inputs[0]->value()->dims();
	_num_samples = inputDim[0];
	_input_height = inputDim[2];
	_input_width = inputDim[3];
	_patch_height = param.patch_height();
	_patch_width = param.patch_width();
	_num_channels = inputDim[1];
	LOG_IF(FATAL, _patch_width > _input_width) << "[FAILED] " << _name << ": Patch width must be less than " << _input_width;
	LOG_IF(FATAL, _patch_height > _input_height) << "[FAILED] " << _name << ": Patch height must be less than " << _input_height;
	_outputs[0]->initValue({ _num_samples, _num_channels, _patch_height, _patch_width });
	_outputs[0]->initDiff();
	_generator.seed(_random_device());
	_patch_max_y = inputDim[2] - _patch_height;
	_patch_max_x = inputDim[3] - _patch_width;	
	_h_x_pos = new int[_num_samples];
	_h_y_pos = new int[_num_samples];
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_x_pos, _num_samples * sizeof(int)));
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_y_pos, _num_samples * sizeof(int)));
}

void PatchSampling::forward()
{	
	std::uniform_int_distribution<int> _x_dist(0, _patch_max_x);
	std::uniform_int_distribution<int> _y_dist(0, _patch_max_y);	
	for (int i = 0; i < _num_samples; ++i) {
		_h_x_pos[i] = _x_dist(_generator);
		_h_y_pos[i] = _y_dist(_generator);		
	}	
	DF_NODE_CUDA_CHECK(cudaMemcpy(_d_x_pos, _h_x_pos, _num_samples * sizeof(int), cudaMemcpyHostToDevice));
	DF_NODE_CUDA_CHECK(cudaMemcpy(_d_y_pos, _h_y_pos, _num_samples * sizeof(int), cudaMemcpyHostToDevice));
	auto size = _outputs[0]->value()->size();
	PatchSamplingForward << < numOfBlocks(size), maxThreadsPerBlock >> > (
		size,
		_num_channels,
		_input_height,
		_input_width,
		_patch_height,
		_patch_width,
		_d_x_pos,
		_d_y_pos,
		(float*)_inputs[0]->value()->data(),
		(float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void PatchSampling::backward()
{
	if (_inputs[0]->diff()) {
		cudaMemset(_inputs[0]->diff()->mutableData(), 0, _inputs[0]->diff()->sizeInBytes());
		auto size = _outputs[0]->value()->size();
		PatchSamplingBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (
			size,
			_num_channels,
			_input_height,
			_input_width,
			_patch_height,
			_patch_width,
			_d_x_pos,
			_d_y_pos,
			(float*)_outputs[0]->diff()->data(),
			(float*)_inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string PatchSampling::to_cpp() const
{
	return std::string();
}

