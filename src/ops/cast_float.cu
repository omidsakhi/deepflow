#include "ops/cast_float.h"

#include "core/common_cu.h"

#include "ops/relu.h"

__global__
void CastFloatKernelForward(int n, const int * __restrict__ x, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = x[i];
}

CastFloat::CastFloat(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_cast_float_param() == false) << "param.has_cast_float_param() == false";
}

void CastFloat::initForward() {
	LOG_IF(FATAL, _inputs[0]->value()->type() != Tensor::Int32) << "Input tensor must be of Int32 type.";	
	_outputs[0]->initValue(_inputs[0]->value()->dims(), Tensor::Float);		
	LOG(INFO) << "Initializing CastFloat " << _name << " - " << _outputs[0]->value()->shape();
}

void CastFloat::initBackward() {
	_outputs[0]->initDiff();
}

void CastFloat::forward() {
	auto size = _inputs[0]->value()->size();
	CastFloatKernelForward << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (int*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();	
}

void CastFloat::backward() {

}