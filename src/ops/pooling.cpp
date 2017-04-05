#include "ops/pooling.h"

Pooling::Pooling(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_op_pooling_param() == false) << "param.has_op_pooling_param() == false";
}

void Pooling::initForward() {
	LOG_IF(FATAL, cudnnCreate(&_cudnnHandle) != 0);
	const OpPoolingParam &param = _param.op_pooling_param();
	LOG_IF(FATAL, param.window_w() < 1);
	LOG_IF(FATAL, param.window_h() < 1);
	LOG_IF(FATAL, param.v_pad() < 0);
	LOG_IF(FATAL, param.h_pad() < 0);
	LOG_IF(FATAL, cudnnCreatePoolingDescriptor(&_poolingDesc) != 0);
	int n, c, h, w;
	LOG_IF(FATAL,cudnnSetPooling2dDescriptor(_poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, param.window_h(), param.window_w(), param.v_pad(), param.h_pad(), param.v_stride(), param.h_stride()) != 0) << "cudnnSetPooling2dDescriptor [FAILED]";
	auto dims = _inputs[0]->dims();
	n = dims[0];
	c = dims[1];
	h = floor((float)dims[2] / param.v_stride());
	w = floor((float)dims[3] / param.h_stride());	
	_outputs[0]->initValue({ n, c, h, w }, Tensor::Float);	
	LOG(INFO) << "Initializing Pooling (name: " << _name << ") | Shape : " << _outputs[0]->value()->toString();	
}

void Pooling::initBackward() {
	_outputs[0]->initDiff();
}

void Pooling::forward() {
	LOG_IF(FATAL, cudnnPoolingForward(_cudnnHandle, _poolingDesc, &_alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &_beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()) != 0) << "cudnnPoolingForward [FAILED]";
}

void Pooling::backward() {
	LOG_IF(FATAL, cudnnPoolingBackward(_cudnnHandle, _poolingDesc, &_alpha, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &_beta, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()) != 0) << "cudnnPoolingBackward [FAILED]";
}
