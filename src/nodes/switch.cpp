#include "nodes\switch.h"

Switch::Switch(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_switch_param() == false) << "param.has_switch_param() == false";
}

int Switch::minNumOutputs()
{
	return m_on ? 1 : 0;
}

std::list<std::shared_ptr<Node>> Switch::outputNodes() const
{
	std::list<std::shared_ptr<Node>> list;
	if (m_on) {
		for (auto node : _outputs[0]->connectedNodes())
			list.push_back(node);
	}	
	return list;
}

std::list<std::shared_ptr<Node>> Switch::inputNodes() const
{
	std::list<std::shared_ptr<Node>> list;
	if (m_on) {		
		list.push_back(_inputs[0]->connectedNode());
	}	
	return list;
}

void Switch::init()
{
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();	
}

void Switch::forward()
{
	if (m_on) {
		cpy(_inputs[0]->value()->size(), 1.0, _inputs[0]->value()->data(), 0.0, _outputs[0]->value()->mutableData());
	}
	else {
		fill(_outputs[0]->value()->size(), 0.0, _outputs[0]->value()->mutableData());
	}
}

void Switch::backward()
{
	if (m_on) {
		cpy(_inputs[0]->diff()->size(), 1.0, _outputs[0]->diff()->data(), 0.0, _inputs[0]->diff()->mutableData());
	}
	else {
		fill(_outputs[0]->diff()->size(), 0.0, _inputs[0]->diff()->mutableData());
	}
}

std::string Switch::to_cpp() const
{
	return std::string();
}

void Switch::setEnabled(bool state)
{
	m_on = state;
}
