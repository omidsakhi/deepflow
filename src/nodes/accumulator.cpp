#include "nodes/accumulator.h"

#include "core/common_cu.h"

Accumulator::Accumulator(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_accumulator_param() == false) << "param.has_accumulator_param() == false";
}

void Accumulator::init() {	
	const deepflow::AccumulatorParam &accParam = _param->accumulator_param();	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	if (_inputs[0]->diff())
		_outputs[0]->initDiff(_inputs[0]->diff()->dims(), _inputs[0]->diff());
	_outputs[1]->initValue({ 1,1,1,1 });
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
}

void Accumulator::forward() {
	auto size = _inputs[0]->value()->size();	
	DF_NODE_CUDNN_CHECK(cudnnAddTensor(_cudnnHandle, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(), &one, _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data()));
	_total += _inputs[0]->dims()[0];	
	DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[1]->value()->gpu_data(), &_total, sizeof(float), cudaMemcpyHostToDevice));
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
	DF_NODE_CUDA_CHECK(cudaMemset(_outputs[0]->value()->gpu_data(), 0, _outputs[0]->value()->bytes()));
	_total = 0;
}
