#include "nodes/equal.h"

__global__
void EqualKernel(int n, const int * __restrict__ a,  const int * __restrict__ b, float * __restrict__ c)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = (a[i] == b[i] ? 1.0 : 0.0);
}


Equal::Equal(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_equal_param() == false) << "param.has_equal_param() == false";
}

void Equal::init() {
	LOG_IF(FATAL, _inputs[0]->value()->type() != Tensor::Int32) << "Inputs must be of type int32.";
	LOG_IF(FATAL, _inputs[1]->value()->type() != Tensor::Int32) << "Inputs must be of type int32.";
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[0]->value()->size()) << "Size mismatch [FAILED]";
	_outputs[0]->initValue(_inputs[0]->value()->dims(), Tensor::TensorType::Float);
	LOG(INFO) << "Equal " << _name << " - " << _outputs[0]->value()->shape();
}

void Equal::forward() {
	auto size = _inputs[0]->value()->size();
	EqualKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (int*)_inputs[0]->value()->data(), (int*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Equal::backward() {
	LOG(FATAL);
}

std::string Equal::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.equal(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
