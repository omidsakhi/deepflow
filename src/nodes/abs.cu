#include "nodes/abs.h"
#include "core/common_cu.h"

__global__
void AbsKernelForward(const int n, const float * __restrict__ x, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		y[i] = fabs(x[i]);
}

__global__
void AbsKernelBackward(const int n, const float *x, const float *diff, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = ((x[i] == 0) ? 0 : (x[i] > 0? diff[i] : - diff[i]));
}

Abs::Abs(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_abs_param() == false) << "param.has_abs_param() == false";
}

void Abs::initForward() {
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Absolute " << _name << " - " << _outputs[0]->value()->shape();
}

void Abs::initBackward() {
	_outputs[0]->initDiff();
}

void Abs::forward() {
	auto size = _inputs[0]->value()->size();
	AbsKernelForward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Abs::backward() {
	auto size = _inputs[0]->value()->size();
	AbsKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_KERNEL_CHECK();
}

std::string Abs::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.square(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
