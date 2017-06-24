#include "nodes/softmax.h"

#include <glog/logging.h>

Softmax::Softmax(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_softmax_param() == false) << "param.has_softmax_param() == false";
}

void Softmax::initForward() {		
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Softmax " << _name << " - " << _outputs[0]->value()->shape();
}

void Softmax::initBackward() {
	_outputs[0]->initDiff();
}

void Softmax::forward() {	
	DF_NODE_CUDNN_CHECK(cudnnSoftmaxForward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));	
}

void Softmax::backward() {		
	DF_NODE_CUDNN_CHECK(cudnnSoftmaxBackward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), &one, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));
}

std::string Softmax::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.softmax(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}

