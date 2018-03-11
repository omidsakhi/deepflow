#include "nodes\pass_through.h"

PassThrough::PassThrough(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_pass_through_param() == false) << "param.has_pass_through_param() == false";	
	_stop_gradients = param->pass_through_param().stop_gradients();
}

void PassThrough::init()
{
	_outputs[0]->initValue(_inputs[0]->value()->dims(), _inputs[0]->value());
	if (!_stop_gradients)
		_outputs[0]->initDiff(_inputs[0]->value()->dims(),_inputs[0]->diff());
}

void PassThrough::forward()
{
}

void PassThrough::backward()
{
	if (_stop_gradients && _inputs[0]->diff())
		cudaMemset(_inputs[0]->diff()->mutableData(), 0, _inputs[0]->diff()->sizeInBytes());
}

std::string PassThrough::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.pass_through(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
