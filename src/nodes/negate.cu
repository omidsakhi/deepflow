#include "nodes\negate.h"

Negate::Negate(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_negate_param() == false) << "param.has_negate_param() == false";
	_negate_type = _param->negate_param().negate_type();
}

void Negate::initForward()
{
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Negate " << _name << " - " << _outputs[0]->value()->shape();
}

void Negate::initBackward()
{
	_outputs[0]->initDiff();
}

void Negate::forward()
{
	auto size = _inputs[0]->value()->size();
	if (_negate_type != deepflow::ActionType::DIFFS) {		
		cpy(size, -1.0f, _inputs[0]->value()->data(), 0.0f, _outputs[0]->value()->mutableData());
	}
	else {
		cpy(size, 1.0f, _inputs[0]->value()->data(), 0.0f, _outputs[0]->value()->mutableData());
	}
}

void Negate::backward()
{
	auto size = _inputs[0]->value()->size();
	if (_negate_type != deepflow::ActionType::VALUES) {
		cpy(size, -1.0f, _outputs[0]->diff()->data(), 1.0f, _inputs[0]->diff()->mutableData());
	}
	else {
		cpy(size, 1.0f, _outputs[0]->diff()->data(), 1.0f, _inputs[0]->diff()->mutableData());
	}
}

std::string Negate::to_cpp() const
{
	return std::string();
}
