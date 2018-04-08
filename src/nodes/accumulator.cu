#include "nodes/accumulator.h"

#include "core/common_cu.h"

__global__
void AccumulatorKernel(int n, const float * __restrict__ x, float * __restrict__ out1)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		out1[i] += x[i];
}

Accumulator::Accumulator(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_accumulator_param() == false) << "param.has_accumulator_param() == false";
}

void Accumulator::init() {	
	const deepflow::AccumulatorParam &accParam = _param->accumulator_param();	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[1]->initValue({ 1,1,1,1 });	
}

void Accumulator::forward() {
	auto size = _inputs[0]->value()->size();
	AccumulatorKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
	_total += _inputs[0]->dims()[0];	
	DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[1]->value()->mutableData(), &_total, sizeof(float), cudaMemcpyHostToDevice));
}

void Accumulator::backward() {
	
}

std::string Accumulator::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.accumulator(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}

void Accumulator::reset()
{
	DF_NODE_CUDA_CHECK(cudaMemset(_outputs[0]->value()->mutableData(), 0, _outputs[0]->value()->sizeInBytes()));
	_total = 0;
}
