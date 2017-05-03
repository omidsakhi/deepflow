#include "nodes/softmax.h"

#include <glog/logging.h>

Softmax::Softmax(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_softmax_param() == false) << "param.has_softmax_param() == false";
}

void Softmax::initForward() {		
	_alpha = _param.softmax_param().alpha();
	_beta = _param.softmax_param().beta();		
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Softmax " << _name << " - " << _outputs[0]->value()->shape();
}

void Softmax::initBackward() {
	_outputs[0]->initDiff();
}

void Softmax::forward() {	
	DF_CUDNN_CHECK(cudnnSoftmaxForward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &_alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &_beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));	
}

void Softmax::backward() {		
	DF_CUDNN_CHECK(cudnnSoftmaxBackward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &_alpha, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), &_beta, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));	
}

std::string Softmax::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.softmax(" + _inputs[0]->connectedNode()->name() + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}

