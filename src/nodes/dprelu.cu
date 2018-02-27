#include "nodes/dprelu.h"

#include "core/common_cu.h"

__global__
void DPReluForwardKernel(int n, int channels, int inner_dims, const float * __restrict__ x, const float * __restrict__ w_l, const float * __restrict__ w_r, float * __restrict__ y)
{
	extern __shared__ float we[];

	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++) {
			we[2 * k] = w_l[k];
			we[2 * k + 1] = w_r[k];
		}
	}
	
	__syncthreads();

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		int iw = (i / inner_dims) % channels;
		if (x[i] > 0)
			y[i] = x[i] * we[2 * iw];
		else {			
			y[i] = x[i] * we[2 * iw + 1];
		}
	}
}

__global__
void DPReluBackwardKernel(int n, int channels, int inner_dims, const float *x, const float *w_l, const float *w_r, const float * dy, float *dx)
{
	extern __shared__ float we[];

	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++) {
			we[2 * k] = w_l[k];
			we[2 * k + 1] = w_r[k];
		}		
	}
	
	__syncthreads();
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		int iw = (i / inner_dims) % channels;
		if (x[i] > 0)
			dx[i] = dy[i] * we[2 * iw];
		else {			
			dx[i] = dy[i] * we[2 * iw + 1];
		}
	}
}

__global__
void DPReluBackwardWeightKernel(int n, int channels, int inner_dims, const float *x, const float * dy, float *dw_l, float *dw_r)
{
	extern __shared__ float sum[];
	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++) {
			sum[2 * k] = 0;
			sum[2 * k + 1] = 0;
			dw_l[k] = 0;
			dw_r[k] = 0;
		}
	}
	
	__syncthreads();
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		int iw = (i / inner_dims) % channels;
		if (x[i] > 0) {
			sum[2 * iw] += dy[i] * x[i] / inner_dims;
		}
		else {
			sum[2 * iw + 1] += dy[i] * x[i] / inner_dims;
		}				
	}
	
	__syncthreads();

	if (threadIdx.x == 0) {
		for (int k = 0; k < channels; k++) {
			atomicAdd(&dw_l[k], sum[2 * k]);
			atomicAdd(&dw_r[k], sum[2 * k + 1]);
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
	DPReluForwardKernel << < numOfBlocks(size), maxThreadsPerBlock, 2 * channels * sizeof(float) >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_inputs[2]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

void DPRelu::backward()
{
	auto size = _inputs[0]->value()->size();
	auto channels = _inputs[0]->dims()[1];
	auto inner_dims = _inputs[0]->dims()[2] * _inputs[0]->dims()[3];
	DPReluBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock, 2 * channels * sizeof(float) >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_inputs[2]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
	DPReluBackwardWeightKernel << < numOfBlocks(size), maxThreadsPerBlock, 2 * channels * sizeof(float) >> >(size, channels, inner_dims, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData(), (float*)_inputs[2]->diff()->mutableData());
	DF_NODE_KERNEL_CHECK();
}

std::string DPRelu::to_cpp() const
{
	return std::string();
}
