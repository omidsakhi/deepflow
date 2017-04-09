#include "core/phaseplexer.h"

Phaseplexer::Phaseplexer(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_phaseplexer_param() == false) << "param.has_phaseplexer_param() == false";
}

void Phaseplexer::initForward()
{
	const PhaseplexerParam &param = _param.phaseplexer_param();
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

void Phaseplexer::forward()
{	
	auto input = _map[_context->phase];
	if (_context->debug_level == 4)
		LOG(INFO) << "FWRD - input: " << input->connectedTerminal()->name();
	LOG_IF(FATAL, cudaMemcpy(_outputs[0]->value()->mutableData(), input->value()->data(), input->value()->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0);
}

void Phaseplexer::backward()
{	
	auto input = _map[_context->phase];
	if (_context->debug_level == 4)
		LOG(INFO) << "BWRD: input: " << input->connectedTerminal()->name();
	LOG_IF(FATAL, cudaMemcpy(input->diff()->mutableData(), _outputs[0]->diff()->data(), _outputs[0]->diff()->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0);
}
