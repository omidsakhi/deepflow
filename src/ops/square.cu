#include "ops/square.h"
#include "core/common_cu.h"

__global__
void SquareKernelForward(const int n, const float * __restrict__ x, float * __restrict__ x2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) x2[i] = x[i] * x[i];
}

__global__
void SquareKernelBackward(const int n, const float *x, const float *diff, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) out[i] = 2.0f * x[i] * diff[i];
}

Square::Square(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_square_param() == false);
}

void Square::initForward() {		
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Square (name: " << _name << " ) | Shape : " << _outputs[0]->value()->shape();
}

void Square::initBackward() {
	_outputs[0]->initDiff();
}

void Square::forward() {	
	auto size = _inputs[0]->value()->size();
	SquareKernelForward <<< numOfBlocks(size), maxThreadsPerBlock >>> (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);
}

void Square::backward() {	
	auto size = _inputs[0]->value()->size();
	SquareKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);	
}