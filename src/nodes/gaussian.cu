#include "nodes/gaussian.h"

__global__
void GaussianKernelForward(const int n, const float *mean, const float *sigma, const float *normal, float * out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = normal[i] * sigma[i] + mean[i];
	}
}

__global__
void GaussianKernelMeanBackward(const int n, const float *dy, float *dmean)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		dmean[i] = dy[i];		
	}
}

__global__
void GaussianKernelSigmaBackward(const int n, const float *normal, const float *dy, float *dsigma)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		dsigma[i] = dy[i] * normal[i];
	}
}

Gaussian::Gaussian(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_gaussian_param() == false) << "param.has_gaussian_param() == false";
}

void Gaussian::init()
{	
	LOG_IF(FATAL, _inputs[0]->value()->dims() != _inputs[1]->value()->dims()) << "mean and sigma must have the same shape.";
	_size = _inputs[0]->value()->size();
	cudaMalloc(&_normal, _inputs[0]->value()->sizeInBytes());
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
}

void Gaussian::forward()
{
	// sample a distribution with mean 0.0 and sigma 1.0
	_mt19937.seed(_random_device());
	std::normal_distribution<float> dist(0.0f, 1.0f);
	float *h_rand = new float[_size];
	for (int i = 0; i < _size; ++i)
		h_rand[i] = dist(_mt19937);
	DF_NODE_CUDA_CHECK(cudaMemcpy(_normal, h_rand, _size * sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_rand;
	GaussianKernelForward << < numOfBlocks(_size), maxThreadsPerBlock >> > (_size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), _normal, (float*)_outputs[0]->value()->mutableData());
}

void Gaussian::backward()
{
	if (_inputs[0]->diff()) {
		GaussianKernelMeanBackward << < numOfBlocks(_size), maxThreadsPerBlock >> > (_size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
		DF_NODE_KERNEL_CHECK();
	}
	if (_inputs[1]->diff()) {
		GaussianKernelSigmaBackward << < numOfBlocks(_size), maxThreadsPerBlock >> > (_size, _normal, (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData());
		DF_NODE_KERNEL_CHECK();
	}
}

std::string Gaussian::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.gaussian(";
	cpp += _input_name_for_cpp(0) + ", ";
	cpp += _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
