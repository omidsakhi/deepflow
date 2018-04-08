
#include "core/common_cu.h"

#include "nodes/square_error.h"

__global__
void SquareErrorForwardKernel(const int n, const float * __restrict__ x1, const float * __restrict__ x2, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float tmp = x1[i] - x2[i];
		y[i] = tmp * tmp;
	}
}

__global__
void SquareErrorBackwardKernel(const int n, const float sign, const float * __restrict__ x1, const float * __restrict__ x2, const float * __restrict__ d, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		y[i] = sign * (x1[i] - x2[i]) * d[i];
}

SquareError::SquareError(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_square_error_param() == false) << "param.has_square_error_param() == false";
}

void SquareError::init() {	
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input " << _inputs[0]->value()->shape() << " != " << " Target " << _inputs[1]->value()->shape();	
	_outputs[0]->initValue(_inputs[0]->dims());
	_outputs[0]->initDiff();	
}

void SquareError::forward() {
	auto size = _inputs[0]->value()->size();
	SquareErrorForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void SquareError::backward() {
	auto size = _inputs[0]->value()->size();
	if (_inputs[0]->diff()) {
		SquareErrorBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, 1.0f, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data() , (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
	if (_inputs[1]->diff()) {
		SquareErrorBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, -1.0f, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data() , (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string SquareError::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.square_error(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
