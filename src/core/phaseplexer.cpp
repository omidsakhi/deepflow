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
	_outputs[0]->initValue(_inputs[0]->dims());
}

void Phaseplexer::initBackward()
{
	_outputs[0]->initDiff();
}

void Phaseplexer::forward()
{
	LOG_IF(FATAL, _executionPhase.empty()) << "Phaseplexer requires phase to be set.";
	auto input = _map[_executionPhase];
	LOG_IF(FATAL, cudaMemcpy(_outputs[0]->value()->mutableData(), input->value()->data(), input->value()->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0);
}

void Phaseplexer::backward()
{
	LOG_IF(FATAL, _executionPhase.empty()) << "Phaseplexer requires phase to be set.";
	auto input = _map[_executionPhase];
	LOG_IF(FATAL, cudaMemcpy(input->diff()->mutableData(), _outputs[0]->diff()->data(), _outputs[0]->diff()->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0);
}
