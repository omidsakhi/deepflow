#include "nodes\lrn.h"

LRN::LRN(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_lrn_param() == false) << "param.has_lrn_param() == false";
}

void LRN::init()
{
	auto param = _param->lrn_param();
	DF_NODE_CUDNN_CHECK(cudnnCreateLRNDescriptor(&_normDesc));
	DF_NODE_CUDNN_CHECK(cudnnSetLRNDescriptor(_normDesc, param.n(), param.alpha(), param.beta(), param.k()));
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
	LOG(INFO) << "LRN " << _name << " - " << _outputs[0]->value()->shape();
}

void LRN::forward()
{
	DF_NODE_CUDNN_CHECK(
		cudnnLRNCrossChannelForward(_cudnnHandle, _normDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(),
			&zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}

void LRN::backward()
{
	DF_NODE_CUDNN_CHECK(
	cudnnLRNCrossChannelBackward(_cudnnHandle, _normDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one,
		_outputs[0]->value()->descriptor(), _outputs[0]->value()->data(),
		_outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(),
		_inputs[0]->value()->descriptor(), _inputs[0]->value()->data(),
		&zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));
}

std::string LRN::to_cpp() const
{
	return std::string();
}
