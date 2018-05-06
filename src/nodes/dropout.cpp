#include "core/common_cu.h"

#include "nodes/dropout.h"

Dropout::Dropout(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_dropout_param() == false) << "param.has_dropout_param() == false";
}

void Dropout::init() {	
	_outputs[0]->initValue(_inputs[0]->value()->dims());	
	_dropout = _param->dropout_param().dropout();	
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	DF_NODE_CUDNN_CHECK(cudnnCreateDropoutDescriptor(&_dropoutDesc));
	DF_NODE_CUDNN_CHECK(cudnnDropoutGetStatesSize(_cudnnHandle, &_state_sizes_in_bytes));
	DF_NODE_CUDA_CHECK(cudaMalloc(&d_states, _state_sizes_in_bytes));
	DF_NODE_CUDNN_CHECK(cudnnSetDropoutDescriptor(_dropoutDesc, _cudnnHandle, _dropout, d_states, _state_sizes_in_bytes, clock()));
	DF_NODE_CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(_inputs[0]->value()->descriptor(), &_reserve_sizes_in_bytes));
	DF_NODE_CUDA_CHECK(cudaMalloc(&d_reserve, _reserve_sizes_in_bytes));	
	_outputs[0]->initDiff();	
}

void Dropout::forward() {
	if (_context->execution_mode == ExecutionContext::TRAIN) {
		DF_NODE_CUDNN_CHECK(cudnnDropoutForward(_cudnnHandle, _dropoutDesc, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(DF_LINE), _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data(DF_LINE), d_reserve, _reserve_sizes_in_bytes));
	}
	else {
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->gpu_data(DF_LINE), _inputs[0]->value()->gpu_data(DF_LINE), _inputs[0]->value()->bytes(), cudaMemcpyDeviceToDevice));
	}	
}

void Dropout::backward() {
	if (_inputs[0]->diff()) {
		if (_context->execution_mode == ExecutionContext::TRAIN) {
			DF_NODE_CUDNN_CHECK(cudnnDropoutBackward(_cudnnHandle, _dropoutDesc, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->gpu_data(DF_LINE), _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->gpu_data(DF_LINE), d_reserve, _reserve_sizes_in_bytes));
		}
		else {
			DF_NODE_CUDA_CHECK(cudaMemcpy(_inputs[0]->diff()->gpu_data(DF_LINE), _outputs[0]->diff()->gpu_data(DF_LINE), _inputs[0]->diff()->bytes(), cudaMemcpyDeviceToDevice));
		}		
	}
}

std::string Dropout::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.dropout(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(_dropout) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
