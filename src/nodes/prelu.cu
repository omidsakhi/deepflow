#include "core/common_cu.h"

#include "nodes/prelu.h"

__global__
void PReluForwardKernel(int n, int channels, int inner_dims, const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y)
{
	extern __shared__ float we[];
	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++)
			we[k] = w[k];
	}
	__syncthreads();
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < n)
	{
		if (x[i] > 0)
			y[i] = x[i];
		else {
			int iw = (i / inner_dims) % channels;
			y[i] = x[i] * we[iw];
		}
	}
}

__global__
void PReluBackwardKernel(int n, int channels, int inner_dims, const float *x, const float *w, const float * dy, float *dx)
{
	extern __shared__ float we[];
	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++)
			we[k] = w[k];
	}
	__syncthreads();
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		if (x[i] > 0)
			dx[i] = dy[i];
		else {
			int iw = (i / inner_dims) % channels;
			dx[i] = dy[i] * we[iw];
		}
	}
}

__global__
void PReluBackwardWeightKernel(int n, int channels, int inner_dims, const float *x, const float *w, const float * dy, float *dw)
{
	extern __shared__ float sum[];
	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++) {
			sum[k] = 0;
			dw[k] = 0;
		}			
	}
	__syncthreads();
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n && x[i] < 0)
	{
		int iw = (i / inner_dims) % channels;
		sum[iw] += dy[i] * x[i] / inner_dims;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++)			
			atomicAdd(&dw[k], sum[k] );
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
	PReluForwardKernel << < numOfBlocks(size), maxThreadsPerBlock, channels * sizeof(float) >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

void PRelu::backward()
{
	auto size = _inputs[0]->value()->size();
	auto channels = _inputs[0]->dims()[1];
	auto inner_dims = _inputs[0]->dims()[2] * _inputs[0]->dims()[3];
	PReluBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock, channels * sizeof(float) >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();	
	PReluBackwardWeightKernel << < numOfBlocks(size), maxThreadsPerBlock, channels * sizeof(float) >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

std::string PRelu::to_cpp() const
{
	return std::string();
}
