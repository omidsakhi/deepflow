#include "nodes\pass_through.h"

PassThrough::PassThrough(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_pass_through_param() == false) << "param.has_pass_through_param() == false";	
}

void PassThrough::init()
{
	_outputs[0]->initValue(_inputs[0]->value()->dims(), _inputs[0]->value());
}

void PassThrough::forward()
{
}

void PassThrough::backward()
{
	cudaMemset(_inputs[0]->diff()->mutableData(), 0, _inputs[0]->diff()->sizeInBytes());
}

std::string PassThrough::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.pass_through(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
