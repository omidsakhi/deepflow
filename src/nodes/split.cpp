#include "nodes/split.h"

Split::Split(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_split_param() == false) << "param.has_split_param() == false";
}

void Split::init()
{
	auto input_dims  = _inputs[0]->value()->dims();
	_outputs[0]->initValue(input_dims);
	_outputs[1]->initValue(input_dims);
	_outputs[0]->initDiff();
	_outputs[1]->initDiff();
	m_size_in_bytes = _inputs[0]->value()->sizeInBytes();
	LOG(INFO) << "Split " << _name << " - 2 x " << _outputs[0]->value()->shape();
}

void Split::forward()
{	
	cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[0]->value()->data(), m_size_in_bytes, cudaMemcpyDeviceToDevice);
	cudaMemcpy(_outputs[1]->value()->mutableData(), _inputs[0]->value()->data(), m_size_in_bytes, cudaMemcpyDeviceToDevice);
}

void Split::backward()
{	
	auto size = _inputs[0]->value()->size();
	if (_inputs[0] && _inputs[0]->diff()) {
		cudaMemcpy(_inputs[0]->diff()->mutableData(), _outputs[0]->diff()->data(), m_size_in_bytes, cudaMemcpyDeviceToDevice);		
		cpy(size, 1.0, _outputs[1]->diff()->data(), 1.0f, _inputs[0]->diff()->mutableData());
	}
}

std::string Split::to_cpp() const
{
	return std::string();
}
