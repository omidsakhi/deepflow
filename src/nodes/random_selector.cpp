#include "nodes/random_selector.h"

RandomSelector::RandomSelector(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_random_selector_param() == false) << "param.has_random_selector_param() == false";
}

void RandomSelector::initForward()
{
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Size mismatch " << _inputs[0]->value()->shape() << " vs " << _inputs[1]->value()->shape();
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
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[0]->value()->data(), _inputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToDevice))
	else
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[1]->value()->data(), _inputs[1]->value()->sizeInBytes(), cudaMemcpyDeviceToDevice))
}

void RandomSelector::backward()
{
		cpy(_outputs[0]->diff()->size(), 1, _outputs[0]->diff()->data(), 1, _inputs[ _selection == 0 ? 0 : 1 ]->diff()->mutableData());
}

std::string RandomSelector::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.random_selector(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += std::to_string(_probability) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
