#include "core/common_cu.h"

#include "nodes/prelu.h"

__global__
void PReluForwardKernel(int n, int channels, int inner_dims, const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < n)
	{
		if (x[i] > 0) {
			y[i] = x[i];
		}
		else 
		{
			const int iw = (i / inner_dims) % channels;
			y[i] = x[i] * w[iw];
		}
	}
}

__global__
void PReluBackwardKernel(int n, int channels, int inner_dims, const float *x, const float *w, const float * dy, float *dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		if (x[i] > 0)
			dx[i] = dy[i];
		else {
			const int iw = (i / inner_dims) % channels;
			dx[i] = dy[i] * w[iw];
		}
	}
}

__global__
void PReluBackwardWeightKernel(int n, int channels, int inner_dims, const float *x, const float *w, const float * dy, float *dw)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n && x[i] < 0)
	{
		int iw = (i / inner_dims) % channels;		
		atomicAdd(&dw[iw], dy[i] * x[i] * channels / n);
	}
}

PRelu::PRelu(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_prelu_param() == false) << "param.has_prelu_param() == false";
}

void PRelu::init()
{
	auto wdims = _inputs[1]->dims();
	auto idims = _inputs[0]->dims();
	LOG_IF(FATAL, wdims[0] != 1 || wdims[1] != idims[1] || wdims[2] != 1 || wdims[3] != 1) << "PRELU " << _name << " weights must be " << "1x" << idims[1] << "x1x1";
	_outputs[0]->initValue(idims);
	_outputs[0]->initDiff();	
}

void PRelu::forward()
{
	auto size = _inputs[0]->value()->size();
	auto channels = _inputs[0]->dims()[1];
	auto inner_dims = _inputs[0]->dims()[2] * _inputs[0]->dims()[3];
	PReluForwardKernel << < numOfBlocks(size), maxThreadsPerBlock>> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

void PRelu::backward()
{
	auto size = _inputs[0]->value()->size();
	auto channels = _inputs[0]->dims()[1];
	auto inner_dims = _inputs[0]->dims()[2] * _inputs[0]->dims()[3];
	PReluBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock>> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
	DF_CUDA_CHECK(cudaMemset(_inputs[1]->diff()->mutableData(), 0, _inputs[1]->diff()->sizeInBytes()));
	PReluBackwardWeightKernel << < numOfBlocks(size), maxThreadsPerBlock>> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

std::string PRelu::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.prelu(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
