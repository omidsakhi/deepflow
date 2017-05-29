#include "nodes/phaseplexer.h"

Phaseplexer::Phaseplexer(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_phaseplexer_param() == false) << "param.has_phaseplexer_param() == false";
}

void Phaseplexer::initForward()
{
	auto param = _param.phaseplexer_param();
	for (int i = 0; i < param.phase_size(); ++i)
		_map.insert(std::pair<std::string, NodeInputPtr>(param.phase(i), _inputs[i]));
	LOG_IF(FATAL, _map.size() < 2);
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size());
	_outputs[0]->initValue(_inputs[0]->dims());
	LOG(INFO) << "Initializing Phaseplexer " << _name << " - " << _outputs[0]->value()->shape();
}

void Phaseplexer::initBackward()
{
	_outputs[0]->initDiff();
}

std::list<std::shared_ptr<Node>> Phaseplexer::inputNodes() const {
	std::list<std::shared_ptr<Node>> list;
	if (_context) {
		auto input = _map.find(_context->phase);
		list.push_back(input->second->connectedNode());
		return list;
	}
	else {
		return Node::inputNodes();
	}	
}

std::string Phaseplexer::to_cpp() const
{
	auto param = _param.phaseplexer_param();
	std::string cpp = "auto " + _name + " = df.phaseplexer(" + _input_name_for_cpp(0) + ", \"" + param.phase(0) + "\", ";
	cpp += _input_name_for_cpp(1) + ", \"" + param.phase(1) + "\", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}

void Phaseplexer::forward()
{	
	auto input = _map.find(_context->phase);
	DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), input->second->value()->data(), input->second->value()->sizeInBytes(), cudaMemcpyDeviceToDevice))	
}

void Phaseplexer::backward()
{	
	auto input = _map.find(_context->phase);
	cpy(_outputs[0]->diff()->size(), 1, _outputs[0]->diff()->data(), 1,  input->second->diff()->mutableData());	
}
