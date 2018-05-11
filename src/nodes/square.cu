#include "nodes/square.h"
#include "core/common_cu.h"

__global__
void SquareKernelForward(const int n, const float * __restrict__ x, float * __restrict__ x2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) 
		x2[i] = x[i] * x[i];
}

__global__
void SquareKernelBackward(const int n, const float *x, const float *diff, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) 
		out[i] = 2.0f * x[i] * diff[i];
}

Square::Square(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_square_param() == false) << "param.has_square_param() == false";
}

void Square::init() {		
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();	
}

void Square::forward() {	
	auto size = _inputs[0]->value()->size();
	SquareKernelForward <<< numOfBlocks(size), maxThreadsPerBlock >>> (size, _inputs[0]->value()->gpu_data(), (float*)_outputs[0]->value()->gpu_data());
	DF_KERNEL_CHECK();
}

void Square::backward() {
	if (_inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();
		SquareKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[0]->value()->gpu_data(), _outputs[0]->diff()->gpu_data(), (float*)_inputs[0]->diff()->gpu_data());
		DF_KERNEL_CHECK();
	}
}

std::string Square::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.square(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
