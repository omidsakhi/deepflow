#include "core/common_cu.h"

#include "nodes/dropout.h"

Dropout::Dropout(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_dropout_param() == false) << "param.has_dropout_param() == false";
}

void Dropout::initForward() {	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Dropout " << _name << " - " << _outputs[0]->value()->shape();
	_dropout = _param.dropout_param().dropout();	
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	DF_CUDNN_CHECK(cudnnCreateDropoutDescriptor(&_dropoutDesc));
	DF_CUDNN_CHECK(cudnnDropoutGetStatesSize(_cudnnHandle, &_state_sizes_in_bytes));
	DF_CUDA_CHECK(cudaMalloc(&d_states, _state_sizes_in_bytes));
	DF_CUDNN_CHECK(cudnnSetDropoutDescriptor(_dropoutDesc, _cudnnHandle, _dropout, d_states, _state_sizes_in_bytes, clock()));
	DF_CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(_inputs[0]->value()->descriptor(), &_reserve_sizes_in_bytes));
	DF_CUDA_CHECK(cudaMalloc(&d_reserve, _reserve_sizes_in_bytes));	
}

void Dropout::initBackward() {
	_outputs[0]->initDiff();
}

void Dropout::forward() {	
	DF_CUDNN_CHECK(cudnnDropoutForward(_cudnnHandle, _dropoutDesc, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData(), d_reserve, _reserve_sizes_in_bytes));	
}

void Dropout::backward() {	
	DF_CUDNN_CHECK(cudnnDropoutBackward(_cudnnHandle, _dropoutDesc, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData(), d_reserve, _reserve_sizes_in_bytes));	
}

std::string Dropout::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.dropout(" + _inputs[0]->connectedNode()->name() + ", ";
	cpp += std::to_string(_dropout) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
