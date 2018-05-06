#include "nodes/equal.h"

__global__
void EqualKernel(int n, const float * __restrict__ a,  const float * __restrict__ b, float * __restrict__ c)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = (llroundf(a[i]) == llroundf (b[i]) ? 1.0 : 0.0);
}


Equal::Equal(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_equal_param() == false) << "param.has_equal_param() == false";
}

void Equal::init() {
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[0]->value()->size()) << "Size mismatch [FAILED]";
	_outputs[0]->initValue(_inputs[0]->value()->dims());	
}

void Equal::forward() {
	auto size = _inputs[0]->value()->size();
	EqualKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, _inputs[0]->value()->gpu_data(DF_LINE), _inputs[1]->value()->gpu_data(DF_LINE), (float*)_outputs[0]->value()->gpu_data(DF_LINE));
	DF_KERNEL_CHECK();
}

void Equal::backward() {
	
}

std::string Equal::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.equal(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
