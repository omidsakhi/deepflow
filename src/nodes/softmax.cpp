#include "nodes/softmax.h"

#include <glog/logging.h>

Softmax::Softmax(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_softmax_param() == false) << "param.has_softmax_param() == false";
}

void Softmax::init() {		
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
	if (_param->softmax_param().mode() == deepflow::SoftmaxParam_Mode_INSTANCE)
		_mode = CUDNN_SOFTMAX_MODE_INSTANCE;
	else
		_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
}

void Softmax::forward() {	
	DF_NODE_CUDNN_CHECK(cudnnSoftmaxForward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, _mode, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data()));
}

void Softmax::backward() {
	if (_inputs[0]->diff())
		DF_NODE_CUDNN_CHECK(cudnnSoftmaxBackward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, _mode, &one, _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->gpu_data(), &zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->gpu_data()));
}

std::string Softmax::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.softmax(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}

