#include "ops/softmax.h"

#include <glog/logging.h>

Softmax::Softmax(NodeParam param) : Node(param) {
	LOG_IF(FATAL, param.has_op_softmax_param() == false);
}

void Softmax::initForward() {		
	_alpha = _param.op_softmax_param().alpha();
	_beta = _param.op_softmax_param().beta();		
	LOG_IF(FATAL, cudnnCreate(&_cudnnHandle) != 0);
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Softmax (name: " << _name << " ) | Shape : " << _outputs[0]->value()->toString();
}

void Softmax::initBackward() {
	_outputs[0]->initDiff();
}

void Softmax::forward() {	
	LOG_IF(FATAL, cudnnSoftmaxForward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &_alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &_beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()) != 0);
}

void Softmax::backward() {		
	LOG_IF(FATAL, cudnnSoftmaxBackward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &_alpha, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), &_beta, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()) != 0);
}

