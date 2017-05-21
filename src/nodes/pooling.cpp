#include "nodes/pooling.h"

Pooling::Pooling(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_pooling_param() == false) << "param.has_pooling_param() == false";
}

void Pooling::initForward() {
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	auto param = _param.pooling_param();
	LOG_IF(FATAL, param.window_w() < 1);
	LOG_IF(FATAL, param.window_h() < 1);
	LOG_IF(FATAL, param.v_pad() < 0);
	LOG_IF(FATAL, param.h_pad() < 0);
	DF_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&_poolingDesc));
	int n, c, h, w;
	DF_CUDNN_CHECK(cudnnSetPooling2dDescriptor(_poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, param.window_h(), param.window_w(), param.v_pad(), param.h_pad(), param.v_stride(), param.h_stride()));
	auto dims = _inputs[0]->dims();
	n = dims[0];
	c = dims[1];
	h = floor((float)dims[2] / param.v_stride());
	w = floor((float)dims[3] / param.h_stride());	
	_outputs[0]->initValue({ n, c, h, w }, Tensor::Float);	
	LOG(INFO) << "Initializing Pooling " << _name << " - " << _inputs[0]->value()->shape() << " -> " << _outputs[0]->value()->shape();	
}

void Pooling::initBackward() {
	_outputs[0]->initDiff();
}

void Pooling::forward() {
	DF_CUDNN_CHECK(cudnnPoolingForward(_cudnnHandle, _poolingDesc, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}

void Pooling::backward() {
	DF_CUDNN_CHECK(cudnnPoolingBackward(_cudnnHandle, _poolingDesc, &one, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &one, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));
}

std::string Pooling::to_cpp() const
{
	auto param = _param.pooling_param();	
	std::string cpp = "auto " + _name + " = df.pooling(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(param.window_h()) + ", ";
	cpp += std::to_string(param.window_w()) + ", ";
	cpp += std::to_string(param.v_pad()) + ", ";
	cpp += std::to_string(param.h_pad()) + ", ";
	cpp += std::to_string(param.v_stride()) + ", ";
	cpp += std::to_string(param.h_stride()) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
