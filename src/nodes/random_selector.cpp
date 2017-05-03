#include "nodes/random_selector.h"

RandomSelector::RandomSelector(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_random_selector_param() == false) << "param.has_random_selector_param() == false";
}

void RandomSelector::initForward()
{
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size());
	_outputs[0]->initValue(_inputs[0]->dims());
	auto param = _param.random_selector_param();
	_probability = param.probability();
	LOG(INFO) << "Initializing RandomSelector " << _name << " - " << _outputs[0]->value()->shape();
}

void RandomSelector::initBackward()
{
	_outputs[0]->initDiff();
}

void RandomSelector::forward()
{
	float rnd = ((float)rand() / RAND_MAX);
	_selection = (rnd < _probability) ? 0 : 1;
	if (_selection == 0)
		DF_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[0]->value()->data(), _inputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToDevice))
	else
		DF_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[1]->value()->data(), _inputs[1]->value()->sizeInBytes(), cudaMemcpyDeviceToDevice))
}

void RandomSelector::backward()
{
	if (_selection == 0)
		DF_CUDA_CHECK(cudaMemcpy(_inputs[0]->diff()->mutableData(), _outputs[0]->diff()->data(), _outputs[0]->diff()->sizeInBytes(), cudaMemcpyDeviceToDevice))
	else
		DF_CUDA_CHECK(cudaMemcpy(_inputs[1]->diff()->mutableData(), _outputs[0]->diff()->data(), _outputs[0]->diff()->sizeInBytes(), cudaMemcpyDeviceToDevice))
}

std::string RandomSelector::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.random_selector(" + _inputs[0]->connectedNode()->name() + ", " + _inputs[1]->connectedNode()->name() + ", ";
	cpp += std::to_string(_probability) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
