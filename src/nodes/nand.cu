#include "nodes/nand.h"

__global__
void NandForwardKernel(const int n, const float * x1, const float * x2, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] = -1.0f * fmin(x1[i], x2[i]);
	}
}

__global__
void NandBackwardKernel(const int n, const float * x1, const float * x2, const float *dy, float *dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float _x1 = x1[i];
		float _x2 = x2[i];
		dx[i] = _x1 <= _x2 ? -dy[i] : 0;
	}
}

Nand::Nand(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_nand_param() == false) << "param.has_nand_param() == false";
}

void Nand::init()
{
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input1 " << _inputs[0]->value()->shape() << " != " << " Input2 " << _inputs[1]->value()->shape();
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
}

void Nand::forward()
{
	auto size = _inputs[0]->value()->size();
	NandForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[0]->value()->gpu_data(), _inputs[1]->value()->gpu_data(), (float*)_outputs[0]->value()->gpu_data());
	DF_KERNEL_CHECK();
}

void Nand::backward()
{
	auto size = _inputs[0]->value()->size();
	if (_inputs[0]->diff()) {
		NandBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[0]->value()->gpu_data(), _inputs[1]->value()->gpu_data(), _outputs[0]->diff()->gpu_data(), (float*)_inputs[0]->diff()->gpu_data());
		DF_KERNEL_CHECK();
	}
	if (_inputs[1]->diff()) {
		NandBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[1]->value()->gpu_data(), _inputs[0]->value()->gpu_data(), _outputs[0]->diff()->gpu_data(), (float*)_inputs[1]->diff()->gpu_data());
		DF_KERNEL_CHECK();
	}
}

std::string Nand::to_cpp() const
{
	return std::string();
}
