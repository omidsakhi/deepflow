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
		_outputs[i]->initValue(input_dims, _inputs[0]->value());
		_outputs[i]->initDiff(true);
	}
}

void Split::forward()
{	
}

void Split::backward()
{	
	if (_inputs[0] && _inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();		
		cudaMemset(_inputs[0]->diff()->mutableData(), 0, size * sizeof(float));
		float alpha = 1.0f / _num_outputs;
		for (int i = 0; i < _num_outputs; ++i) {
			cpy(size, alpha, _outputs[i]->diff()->data(), 1.0f, _inputs[0]->diff()->mutableData());
		}
	}
}

std::string Split::to_cpp() const
{
	return std::string();
}
