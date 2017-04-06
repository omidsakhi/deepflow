#include "core/common_cu.h"

#include "ops/relu.h"

__global__
void ReluKernel(int n, const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ z, const float slope)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		if (x[i] > 0)
			z[i] = y[i];
		else
			z[i] = y[i] * slope;
	}
}

Relu::Relu(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_op_relu_param() == false) << "param.has_op_relu_param() == false";	
}

void Relu::initForward() {	
	_negative_slope = _param.op_relu_param().negative_slope();
	LOG_IF(FATAL, _negative_slope > 0) << " negative_slope > 0";
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Relu (name: " << _name << " ) | Shape : " << _outputs[0]->value()->toString();
}

void Relu::initBackward() {
	_outputs[0]->initDiff();
}

void Relu::forward() {	
	auto size = _inputs[0]->value()->size();	
	ReluKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_inputs[0]->value()->data(), (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), _negative_slope);
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);	
}

void Relu::backward() {	
	auto size = _inputs[0]->value()->size();	
	ReluKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), _negative_slope);
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);	
}