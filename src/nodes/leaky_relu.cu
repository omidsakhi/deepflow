#include "core/common_cu.h"

#include "nodes/leaky_relu.h"

__global__
void ReluKernel(int n, const float * __restrict__ x, const float * __restrict__ y, float beta, float * __restrict__ z, const float slope)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		if (x[i] > 0)
			z[i] = beta * z[i] + y[i];
		else
			z[i] = beta * z[i] + y[i] * slope;
	}
}

LeakyRelu::LeakyRelu(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_leaky_relu_param() == false) << "param.has_leaky_relu_param() == false";	
}

void LeakyRelu::initForward() {	
	_negative_slope = _param.leaky_relu_param().negative_slope();
	LOG_IF(FATAL, _negative_slope < 0) << " negative_slope < 0";
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing LeakyRelu " << _name << " - " << _outputs[0]->value()->shape();
}

void LeakyRelu::initBackward() {
	_outputs[0]->initDiff();
}

void LeakyRelu::forward() {	
	auto size = _inputs[0]->value()->size();	
	ReluKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_inputs[0]->value()->data(), (float*)_inputs[0]->value()->data(), 0.0f, (float*)_outputs[0]->value()->mutableData(), _negative_slope);
	DF_KERNEL_CHECK();	
}

void LeakyRelu::backward() {	
	auto size = _inputs[0]->value()->size();	
	ReluKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), 1.0f, (float*)_inputs[0]->diff()->mutableData(), _negative_slope);
	DF_KERNEL_CHECK();	
}

std::string LeakyRelu::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.leaky_relu(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(_negative_slope) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
