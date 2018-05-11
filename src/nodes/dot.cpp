#include "nodes/dot.h"

Dot::Dot(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_dot_param() == false) << "param.has_dot_param() == false";
}

void Dot::init() {

	auto a = _inputs[0];
	auto ad = a->dims();

	auto b = _inputs[1];
	auto bd = b->dims();

	LOG_IF(FATAL, a->value()->size() != b->value()->size()) 
		<< "Different sizes " << a->value()->shape() << "(" << a->connectedNode()->name() << ") vs " << b->value()->shape() << " (" << b->connectedNode()->name() << ")";
	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();	
}

void Dot::forward() {	
	auto size = _outputs[0]->value()->size();
	dot(size, 1.0, _inputs[0]->value()->gpu_data(), _inputs[1]->value()->gpu_data(), 0.0, _outputs[0]->value()->gpu_data());	
	DF_KERNEL_CHECK();
}

void Dot::backward() {
	auto size = _outputs[0]->diff()->size();
	if (_inputs[0]->diff()) {
		dot(size, 1.0, _outputs[0]->diff()->gpu_data(), _inputs[1]->value()->gpu_data(), 0.0, _inputs[0]->diff()->gpu_data());
		DF_KERNEL_CHECK();
	}
	if (_inputs[1]->diff()) {
		dot(size, 1.0, _outputs[0]->diff()->gpu_data(), _inputs[0]->value()->gpu_data(), 0.0, _inputs[1]->diff()->gpu_data());
		DF_KERNEL_CHECK();
	}
}

std::string Dot::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.dot(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}