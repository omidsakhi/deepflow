#include "core/common_cu.h"

#include "nodes/bias_add.h"

__global__
void BiasAddKernelForward(const int n, const float *in, const int inner_dim, const int bias_dim, const float *b, float *out)
{
	extern __shared__ float bias[];	
	if (threadIdx.x == 0) {
		for (int k = 0; k < bias_dim; k++)
			bias[k] = b[k];
	}
	__syncthreads();
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int index = (i/inner_dim)%bias_dim;
		out[i] = in[i] + bias[index];
	}
}

__global__
void BiasAddKernelBackward(const int n, const float *diff, const int inner_dim, const int num_samples, const int bias_dim, float *bias_diff)
{
	extern __shared__ float sum[];	
	if (threadIdx.x == 0) {
		for (int k = 0; k < bias_dim; k++) {
			sum[k] = 0;
			bias_diff[k] = 0;
		}
	}
	__syncthreads();
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		int index = (i / inner_dim) % bias_dim;
		sum[index] += diff[i] / inner_dim;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int k = 0; k < bias_dim; k++)
			atomicAdd(&bias_diff[k], sum[k]);
	}
}

BiasAdd::BiasAdd(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_bias_add_param() == false) << "param.has_bias_add_param() == false";
}

void BiasAdd::init() {
	auto inputDim = _inputs[0]->dims();
	_inner_dim = inputDim[2] * inputDim[3];
	auto weightDim = _inputs[1]->dims();
	LOG_IF(FATAL, inputDim[1] != weightDim[1]) << _name << "Bias channels between input and bias weights must be the same.";
	LOG_IF(FATAL, weightDim[0] != 1 || weightDim[2] != 1 || weightDim[3] != 1) << _name << "All bias weight dimentions must be one except channels.";
	_bias_dim = weightDim[1];
	_sample_dim = inputDim[0];
	_outputs[0]->initValue(inputDim);
	_outputs[0]->initDiff(inputDim, _inputs[0]->diff());	
}

void BiasAdd::forward() {
	// C(m,n) = A(m,n) + B(m,n)	
	auto size = _outputs[0]->value()->size();
	BiasAddKernelForward <<< numOfBlocks(size), maxThreadsPerBlock, _bias_dim * sizeof(float) >>> (size, (float*) _inputs[0]->value()->data(), _inner_dim, _bias_dim, (float*) _inputs[1]->value()->data(), (float*) _outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void BiasAdd::backward() {
	auto size = _outputs[0]->diff()->size();		
	BiasAddKernelBackward << < numOfBlocks(size), maxThreadsPerBlock, _bias_dim * sizeof(float) >> > (size, (float*)_outputs[0]->diff()->data(), _inner_dim, _sample_dim, _bias_dim, (float*)_inputs[1]->diff()->mutableData());
	DF_KERNEL_CHECK();
}

std::string BiasAdd::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.bias_add(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
