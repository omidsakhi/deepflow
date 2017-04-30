#include "nodes/accumulator.h"

#include "core/common_cu.h"

__global__
void AccumulatorKernel(int n, const float * __restrict__ x, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] += x[i];
}

Accumulator::Accumulator(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_accumulator_param() == false) << "param.has_accumulator_param() == false";
}

void Accumulator::initForward() {	
	const AccumulatorParam &accParam = _param.accumulator_param();
	_reset_time = (ResetTime)accParam.reset_time();
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Accumulator " << _name << " - " << _outputs[0]->value()->shape();
}

void Accumulator::initBackward() {
	_outputs[0]->initDiff();
}

void Accumulator::forward() {
	if (_reset_time == EndOfEpoch && _context->current_iteration_per_epoch == 1)
		DF_CUDA_CHECK(cudaMemset(_outputs[0]->value()->mutableData(), 0, _outputs[0]->value()->sizeInBytes()));		
	auto size = _inputs[0]->value()->size();
	AccumulatorKernel << < numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Accumulator::backward() {

}

std::string Accumulator::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.accumulator(" + _inputs[0]->connectedNode()->name() + ", ";
	if (_reset_time == EndOfEpoch)
		cpp += "Accumulator::EndOfEpoch, ";
	else if (_reset_time == Never) 
		cpp += "Accumulator::Never, ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
