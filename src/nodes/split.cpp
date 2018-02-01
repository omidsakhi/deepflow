#include "nodes/split.h"

Split::Split(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_split_param() == false) << "param.has_split_param() == false";
}

void Split::initForward()
{
	auto input_dims  = _inputs[0]->value()->dims();
	_outputs[0]->initValue(input_dims);
	_outputs[1]->initValue(input_dims);
}

void Split::initBackward()
{
	_outputs[0]->initDiff();
	_outputs[1]->initDiff();
}

void Split::forward()
{
	m_size_in_bytes = _inputs[0]->value()->sizeInBytes();
	auto size = _inputs[0]->value()->size();
	cpy(size, 1.0, _inputs[0]->value()->data(), 0.0f, _outputs[0]->value()->mutableData());
	cpy(size, 1.0, _inputs[0]->value()->data(), 0.0f, _outputs[1]->value()->mutableData());
}

void Split::backward()
{	
	auto size = _inputs[0]->value()->size();
	DF_CUDA_CHECK(cudaMemset(_inputs[0]->diff()->mutableData(), 0, m_size_in_bytes));
	cpy(size, 1.0, _outputs[0]->diff()->data(), 1.0f, _inputs[0]->diff()->mutableData());
	cpy(size, 1.0, _outputs[1]->diff()->data(), 1.0f, _inputs[0]->diff()->mutableData());
}

std::string Split::to_cpp() const
{
	return std::string();
}
