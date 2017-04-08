#include "core/common_cu.h"

#include "ops/dropout.h"

Dropout::Dropout(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_dropout_param() == false);
}

void Dropout::initForward() {	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Dropout " << _name << " - " << _outputs[0]->value()->shape();
	_dropout = _param.dropout_param().dropout();
	cudnnStatus_t status;
	LOG_IF(FATAL, (status = cudnnCreate(&_cudnnHandle)) != 0) << cudaStatusToString(status);	
	LOG_IF(FATAL, (status = cudnnCreateDropoutDescriptor(&_dropoutDesc)) != 0) << cudaStatusToString(status);
	LOG_IF(FATAL, (status = cudnnDropoutGetStatesSize(_cudnnHandle, &_state_sizes_in_bytes)) != 0) << cudaStatusToString(status);
	LOG_IF(FATAL, cudaMalloc(&d_states, _state_sizes_in_bytes) != 0);
	LOG_IF(FATAL, (status = cudnnSetDropoutDescriptor(_dropoutDesc, _cudnnHandle, _dropout, d_states, _state_sizes_in_bytes, clock())) != 0) << cudaStatusToString(status);
	LOG_IF(FATAL, (status = cudnnDropoutGetReserveSpaceSize(_inputs[0]->value()->descriptor(), &_reserve_sizes_in_bytes)) != 0) << cudaStatusToString(status);
	LOG_IF(FATAL, cudaMalloc(&d_reserve, _reserve_sizes_in_bytes) != 0);
}

void Dropout::initBackward() {
	_outputs[0]->initDiff();
}

void Dropout::forward() {	
	LOG_IF(FATAL, cudnnDropoutForward(_cudnnHandle, _dropoutDesc, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData(), d_reserve, _reserve_sizes_in_bytes) != 0) << "cudnnDropoutForward [FAILED]";
}

void Dropout::backward() {	
	LOG_IF(FATAL, cudnnDropoutBackward(_cudnnHandle, _dropoutDesc, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData(), d_reserve, _reserve_sizes_in_bytes) != 0) << "cudnnDropoutBackward [FAILED]";
}
