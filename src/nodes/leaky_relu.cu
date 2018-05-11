#include "core/common_cu.h"

#include "nodes/leaky_relu.h"

__global__
void ReluKernel(int n, const float *x, const float *x_dy, float *y_dx, const float slope)
{	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{		
		if (x[i] > 0)
			y_dx[i] = x_dy[i];
		else
			y_dx[i] = x_dy[i] * slope;
	}
}

LeakyRelu::LeakyRelu(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_leaky_relu_param() == false) << "param.has_leaky_relu_param() == false";	
}

void LeakyRelu::init() {	
	_initial_negative_slope = _param->leaky_relu_param().negative_slope();	
	LOG_IF(FATAL, _initial_negative_slope < 0) << " negative_slope < 0";
	_negative_slope = _initial_negative_slope;	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();	
}

void LeakyRelu::forward() {	
	auto size = _inputs[0]->value()->size();
	ReluKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, _inputs[0]->value()->gpu_data(), _inputs[0]->value()->gpu_data(), (float*)_outputs[0]->value()->gpu_data(), _negative_slope);
	DF_KERNEL_CHECK();	
}

void LeakyRelu::backward() {
	if (_inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();
		ReluKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _inputs[0]->value()->gpu_data(), _outputs[0]->diff()->gpu_data(), (float*)_inputs[0]->diff()->gpu_data(), _negative_slope);
		DF_KERNEL_CHECK();
	}
}

std::string LeakyRelu::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.leaky_relu(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(_negative_slope) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
