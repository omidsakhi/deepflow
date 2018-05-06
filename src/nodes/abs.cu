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

void Abs::init() {
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();	
}

void Abs::forward() {
	auto size = _inputs[0]->value()->size();
	AbsKernelForward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[0]->value()->gpu_data(DF_LINE), (float*)_outputs[0]->value()->gpu_data(DF_LINE));
	DF_KERNEL_CHECK();
}

void Abs::backward() {
	if (_inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();
		AbsKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[0]->value()->gpu_data(DF_LINE), _outputs[0]->diff()->gpu_data(DF_LINE), (float*)_inputs[0]->diff()->gpu_data(DF_LINE));
		DF_KERNEL_CHECK();
	}
}

std::string Abs::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.square(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
