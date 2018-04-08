#include "nodes/exp.h"
#include "core/common_cu.h"

__global__
void ExpKernelForward(const int n, const float * __restrict__ x, float * __restrict__ x2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		x2[i] = exp(x[i]);
}

__global__
void ExpKernelBackward(const int n, const float *x, const float *diff, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = exp(x[i]) * diff[i];
}

Exp::Exp(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_exp_param() == false) << "param.has_exp_param() == false";
}

void Exp::init() {
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();	
}

void Exp::forward() {
	auto size = _inputs[0]->value()->size();
	ExpKernelForward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Exp::backward() {
	if (_inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();
		ExpKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string Exp::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.exp(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
