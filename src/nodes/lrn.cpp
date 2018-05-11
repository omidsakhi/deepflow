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
}

void LRN::forward()
{
	DF_NODE_CUDNN_CHECK(
		cudnnLRNCrossChannelForward(_cudnnHandle, _normDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(),
			&zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data()));
}

void LRN::backward()
{
	if (_inputs[0]->diff()) {
		/*
		DF_NODE_CUDNN_CHECK(
			cudnnLRNCrossChannelForward(_cudnnHandle, _normDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
				&one, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->gpu_data(),
				&zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->gpu_data()));
		*/

		DF_NODE_CUDNN_CHECK(
			cudnnLRNCrossChannelBackward(_cudnnHandle, _normDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one,
				_outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data(),
				_outputs[0]->diff()->descriptor(), _outputs[0]->diff()->gpu_data(),
				_inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(),
				&zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->gpu_data()));
	}
}

std::string LRN::to_cpp() const
{
	auto param = _param->lrn_param();
	// std::string lrn(std::string input, std::string name = "lrn", int n = 5, float alpha = 1e-4f, float beta = 0.75f, float k = 2.0f, std::initializer_list<std::string> phases = {});
	std::string cpp = "auto " + _name + " = df.lrn(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += std::to_string(param.n()) + ", ";
	cpp += std::to_string(param.alpha()) + ", ";
	cpp += std::to_string(param.beta()) + ", ";
	cpp += std::to_string(param.k()) + ");";	
	return cpp;
}
