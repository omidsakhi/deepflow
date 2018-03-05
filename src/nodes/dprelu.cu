#include "nodes/dprelu.h"

#include "core/common_cu.h"

__global__
void DPReluForwardKernel(int n, int channels, int inner_dims, const float * __restrict__ x, const float * __restrict__ w_l, const float * __restrict__ w_r, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		const int iw = (i / inner_dims) % channels;
		if (x[i] > 0)
			y[i] = x[i] * w_l[iw];
		else {			
			y[i] = x[i] * w_r[iw];
		}
	}
}

__global__
void DPReluBackwardKernel(int n, int channels, int inner_dims, const float *x, const float *w_l, const float *w_r, const float * dy, float *dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		const int iw = (i / inner_dims) % channels;
		if (x[i] > 0)
			dx[i] = dy[i] * w_l[iw];
		else {			
			dx[i] = dy[i] * w_r[iw];
		}
	}
}

__global__
void DPReluBackwardWeightKernel(int n, int channels, int inner_dims, const float *x, const float * dy, float *dw_l, float *dw_r)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		const int iw = (i / inner_dims) % channels;
		float denom = n / channels;
		if (x[i] > 0) {
			atomicAdd(&dw_l[iw], dy[i] * x[i] / denom);
		}
		else {
			atomicAdd(&dw_r[iw], dy[i] * x[i] / denom);
		}				
	}
}

DPRelu::DPRelu(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_dprelu_param() == false) << "param.has_dprelu_param() == false";
}

void DPRelu::init()
{
	auto idims = _inputs[0]->dims();
	auto wdimsl = _inputs[1]->dims();
	auto wdimsr = _inputs[2]->dims();	
	LOG_IF(FATAL, wdimsl[0] != 1 || wdimsl[1] != idims[1] || wdimsl[2] != 1 || wdimsl[3] != 1) << "DPRELU " << _name << " weights must be " << "1x" << idims[1] << "x1x1";
	LOG_IF(FATAL, wdimsr[0] != 1 || wdimsr[1] != idims[1] || wdimsr[2] != 1 || wdimsr[3] != 1) << "DPRELU " << _name << " weights must be " << "1x" << idims[1] << "x1x1";
	_outputs[0]->initValue(idims);
	_outputs[0]->initDiff();
}

void DPRelu::forward()
{
	auto size = _inputs[0]->value()->size();
	auto channels = _inputs[0]->dims()[1];
	auto inner_dims = _inputs[0]->dims()[2] * _inputs[0]->dims()[3];
	DPReluForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_inputs[2]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

void DPRelu::backward()
{
	auto size = _inputs[0]->value()->size();
	auto channels = _inputs[0]->dims()[1];
	auto inner_dims = _inputs[0]->dims()[2] * _inputs[0]->dims()[3];
	DPReluBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_inputs[2]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
	DF_CUDA_CHECK(cudaMemset(_inputs[1]->diff()->mutableData(), 0, _inputs[1]->diff()->sizeInBytes()));
	DF_CUDA_CHECK(cudaMemset(_inputs[2]->diff()->mutableData(), 0, _inputs[2]->diff()->sizeInBytes()));
	DPReluBackwardWeightKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData(), (float*)_inputs[2]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

std::string DPRelu::to_cpp() const
{
	return std::string();
}
