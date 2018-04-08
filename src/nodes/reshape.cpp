#include "nodes\reshape.h"

Reshape::Reshape(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_reshape_param() == false) << "param.has_reshape_param() == false";
}

void Reshape::init()
{
	auto param = _param->reshape_param();
	LOG_IF(FATAL, param.output_dims_size() != 4) << _name << " Output dimension must have exactly 4 elements.";
	for (int i=0; i < 4; ++i)
		_output_dims[i] = param.output_dims(i);
	_outputs[0]->initValue(_output_dims, _inputs[0]->value());
	_outputs[0]->initDiff(_output_dims, _inputs[0]->diff());
}

void Reshape::forward()
{
}

void Reshape::backward()
{
}

std::string Reshape::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.reshape(" + _input_name_for_cpp(0) + ", ";
	cpp += "{" + std::to_string(_output_dims[0]) + ", " + std::to_string(_output_dims[1]) + ", " + std::to_string(_output_dims[2]) + ", " + std::to_string(_output_dims[3]) + "}, ";
	cpp += "\"" + _name + "\");";	
	return cpp;	
}
