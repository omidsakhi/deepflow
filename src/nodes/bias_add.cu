#include "core/common_cu.h"

#include "nodes/bias_add.h"

__global__
void BiasAddKernelForward(const int n, const float *in, const int inner_dim, const int bias_dim, const float *b, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = in[i] + b[(i/inner_dim)%bias_dim];
	}
}

__global__
void BiasAddKernelBackward(const int n, const float *diff, const int inner_dim, const int num_samples, const int bias_dim, float *bias_diff)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		int index = (i/inner_dim)%bias_dim;
		atomicAdd(&bias_diff[index], diff[i] / num_samples);
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
	_outputs[0]->initDiff();
	LOG(INFO) << "Bias " << _name << " - " << _outputs[0]->value()->shape();	
}

void BiasAdd::forward() {
	// C(m,n) = A(m,n) + B(m,n)	
	auto size = _outputs[0]->value()->size();
	BiasAddKernelForward <<< numOfBlocks(size), maxThreadsPerBlock >>> (size, (float*) _inputs[0]->value()->data(), _inner_dim, _bias_dim, (float*) _inputs[1]->value()->data(), (float*) _outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void BiasAdd::backward() {
	if (_inputs[0]->connectedNode()) {
		cpy(_inputs[0]->diff()->size(), 1, _outputs[0]->diff()->data(), 0, _inputs[0]->diff()->mutableData());		
	}
	if (_inputs[1]->connectedNode()) {
		auto size = _outputs[0]->diff()->size();
		DF_CUDA_CHECK(cudaMemset(_inputs[1]->diff()->mutableData(), 0, _inputs[1]->diff()->sizeInBytes()));
		BiasAddKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), _inner_dim, _sample_dim, _bias_dim, (float*)_inputs[1]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string BiasAdd::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.bias_add(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
