#include "core/common_cu.h"

#include "nodes/bias_add.h"

__global__
void BiasAddKernelForward(const int n, const float *a, const int bias_dim, const float *b, float *c)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] + b[i%bias_dim];
}

__global__
void BiasAddKernelBackward(const int n, const float *diff, const int num_samples, const int bias_dim, float *bias_diff)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		int j = i%bias_dim;		
		atomicAdd(&bias_diff[j], diff[i] / num_samples);		
	}
}

BiasAdd::BiasAdd(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_bias_add_param() == false) << "param.has_bias_add_param() == false";
}

void BiasAdd::initForward() {
	auto a = _inputs[0];
	auto ad = a->dims();

	auto b = _inputs[1];
	auto bd = b->dims();

	LOG_IF(FATAL, ad[1] != bd[1]) << "ad[1] != bd[1] [FAILED]";
	_outputs[0]->initValue({ ad[0], ad[1], 1, 1 });	
	LOG(INFO) << "Initializing Bias " << _name << " - " << a->value()->shape() << " + " << b->value()->shape() << " -> " << _outputs[0]->value()->shape();	
}

void BiasAdd::initBackward() {
	_outputs[0]->initDiff();
}

void BiasAdd::forward() {
	// C(m,n) = A(m,n) + B(m,n)	
	auto outputDims = _outputs[0]->value()->dims();	
	auto size = _outputs[0]->value()->size();
	auto bias_dim = outputDims[1];
	BiasAddKernelForward <<< numOfBlocks(size), maxThreadsPerBlock >>> (size, (float*) _inputs[0]->value()->data(), bias_dim, (float*) _inputs[1]->value()->data(), (float*) _outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void BiasAdd::backward() {
	if (_inputs[0]->connectedNode()->propagateBack()) {
		DF_CUDA_CHECK(cudaMemcpy(_inputs[0]->diff()->mutableData(), _outputs[0]->diff()->data(), _inputs[0]->diff()->sizeInBytes(), cudaMemcpyDeviceToDevice));
	}
	if (_inputs[1]->connectedNode()->propagateBack()) {
		DF_CUDA_CHECK(cudaMemset(_inputs[1]->diff()->mutableData(), 0, _inputs[1]->diff()->sizeInBytes()));
		auto outputDims = _outputs[0]->diff()->dims();
		auto size = _outputs[0]->diff()->size();
		BiasAddKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), outputDims[0], outputDims[1], (float*)_inputs[1]->diff()->mutableData());
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
