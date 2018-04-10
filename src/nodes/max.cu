#include "nodes/max.h"

__global__
void MaxForwardKernel(const int n, const float * x1, const float * x2, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float _x1 = x1[i];
		float _x2 = x2[i];
		y[i] = _x1 > _x2 ? _x1 : _x2;
	}
}

__global__
void MaxBackwardKernel(const int n, const float * x1, const float * x2, const float *dy, float *dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float _x1 = x1[i];
		float _x2 = x2[i];
		dx[i] = _x1 >= _x2 ? dy[i] : 0;
	}
}

Max::Max(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_max_param() == false) << "param.has_max_param() == false";
}

void Max::init()
{
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input1 " << _inputs[0]->value()->shape() << " != " << " Input2 " << _inputs[1]->value()->shape();
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
}

void Max::forward()
{
	auto size = _inputs[0]->value()->size();
	MaxForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Max::backward()
{
	auto size = _inputs[0]->value()->size();
	if (_inputs[0]->diff()) {
		MaxBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
	if (_inputs[1]->diff()) {
		MaxBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[1]->value()->data(), (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string Max::to_cpp() const
{
	return std::string();
}
