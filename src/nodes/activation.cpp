#include "core/common_cu.h"
#include "nodes/activation.h"

Activation::Activation(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_activation_param() == false) << "param.has_activation_param() == false";
}

std::string Activation::op_name() const
{
	auto activation_param = _param->activation_param();
	cudnnActivationMode_t _activation_mode = (cudnnActivationMode_t)activation_param.type();
	std::string op;
	switch (_activation_mode) {
	case CUDNN_ACTIVATION_SIGMOID:
		op = "sigmoid";
		break;
	case CUDNN_ACTIVATION_RELU:
		op = "relu";
		break;
	case CUDNN_ACTIVATION_TANH:
		op = "tanh";
		break;
	case CUDNN_ACTIVATION_CLIPPED_RELU:
		op = "clipped_relu";
		break;
	case CUDNN_ACTIVATION_ELU:
		op = "elu";
	};
	return op;
}

void Activation::init()
{	
	auto activation_param = _param->activation_param();
	cudnnActivationMode_t _activation_mode =  (cudnnActivationMode_t) activation_param.type();
	float coef = activation_param.coef();
	DF_NODE_CUDNN_CHECK(cudnnCreateActivationDescriptor(&_activation_desc));
	DF_NODE_CUDNN_CHECK(cudnnSetActivationDescriptor(_activation_desc, _activation_mode, CUDNN_PROPAGATE_NAN, coef));
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
}

void Activation::forward()
{
	DF_NODE_CUDNN_CHECK(cudnnActivationForward(_cudnnHandle, _activation_desc, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));	
}

void Activation::backward()
{
	if (_inputs[0]->diff()) {
		DF_NODE_CUDNN_CHECK(cudnnActivationBackward(_cudnnHandle, _activation_desc, &one, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));		
	}
}

std::string Activation::to_cpp() const
{
	auto activation_param = _param->activation_param();	
	float coef = activation_param.coef();
	std::string cpp = "auto " + _name + " = df."+ op_name() + "(" + _input_name_for_cpp(0) + ", ";
	std::string op = op_name();
	if (op == "clipped_relu" || op == "elu")
		cpp += std::to_string(coef) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
