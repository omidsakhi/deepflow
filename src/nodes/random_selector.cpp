#include "nodes/random_selector.h"

RandomSelector::RandomSelector(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_random_selector_param() == false) << "param.has_random_selector_param() == false";
}

void RandomSelector::initForward()
{
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size());
	_outputs[0]->initValue(_inputs[0]->dims());
	LOG(INFO) << "Initializing RandomSelector " << _name << " - " << _outputs[0]->value()->shape();
}

void RandomSelector::initBackward()
{
	_outputs[0]->initDiff();
}

void RandomSelector::forward()
{
	_selection = (rand() % 2 == 0) ? 0 : 1;
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
