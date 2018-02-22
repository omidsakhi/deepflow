#include "nodes/concate.h"

Concate::Concate(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_concate_param() == false) << "param.has_concate_param() == false";
}

void Concate::init()
{
	auto d1 = _inputs[0]->dims();
	auto d2 = _inputs[1]->dims();
	LOG_IF(FATAL, d1[1] != d2[1] || d1[2] != d2[2] || d1[3] != d2[3]) << "Concate " << _name << " | inputs must have the same number of channels, width and height.";
	_first_input_batches = d1[0];
	_second_input_batches = d2[0];	
	_chw = d1[1] * d1[2] * d1[3];
	_outputs[0]->initValue({ _first_input_batches + _second_input_batches, d1[1], d1[2], d1[3] });
	_outputs[0]->initDiff();
	LOG(INFO) << "Concate " << _name << " - " << _outputs[0]->value()->shape();
}

void Concate::forward()
{
	cpy(_first_input_batches * _chw, 1.0, _inputs[0]->value()->data(), 0.0, _outputs[0]->value()->mutableData());
	float * addr = (float*) _outputs[0]->value()->mutableData();
	cpy(_second_input_batches * _chw, 1.0, _inputs[1]->value()->data(), 0.0,  addr + (_first_input_batches * _chw));
}

void Concate::backward()
{
	if (_inputs[0]->diff())
		cpy(_first_input_batches * _chw, 1.0, _outputs[0]->diff()->data(), 0.0f, _inputs[0]->diff()->mutableData());
	if (_inputs[1]->diff()) {
		float * addr = (float*)_outputs[0]->diff()->data();
		cpy(_second_input_batches * _chw, 1.0, addr + (_first_input_batches * _chw), 0.0f, _inputs[1]->diff()->mutableData());
	}
}

std::string Concate::to_cpp() const
{
	return std::string();
}
