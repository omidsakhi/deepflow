#include "nodes/split.h"

Split::Split(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_split_param() == false) << "param.has_split_param() == false";
	_num_outputs = param->split_param().num_outputs();
}

int Split::minNumOutputs()
{
	return _num_outputs;
}

void Split::init()
{
	auto input_dims  = _inputs[0]->value()->dims();
	for (int i = 0; i < _num_outputs; ++i) {
		_outputs[i]->initValue(input_dims);
		_outputs[i]->initDiff(true);
	}
}

void Split::forward()
{	
	for (int i = 0; i < _num_outputs; ++i) {
		DF_NODE_CUDA_CHECK(
		cudaMemcpy(_outputs[i]->value()->gpu_data(DF_LINE), _inputs[0]->value()->gpu_data(DF_LINE), _inputs[0]->value()->bytes(), cudaMemcpyDeviceToDevice)
		);
	}
}

void Split::backward()
{	
	if (_inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();		
		DF_NODE_CUDA_CHECK(
		cudaMemset(_inputs[0]->diff()->gpu_data(DF_LINE), 0, size * sizeof(float))
		)
		float alpha = 1.0f / _num_outputs;
		for (int i = 0; i < _num_outputs; ++i) {
			cpy(size, alpha, _outputs[i]->diff()->gpu_data(DF_LINE), 1.0f, _inputs[0]->diff()->gpu_data(DF_LINE));
		}
	}
}

std::string Split::to_cpp() const
{
	return std::string();
}
