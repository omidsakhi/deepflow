#include "core/common_cu.h"
#include "nodes/activation.h"

Activation::Activation(const NodeParam & param) : Node(param)
{
	LOG_IF(FATAL, param.has_activation_param() == false) << "param.has_activation_param() == false";
}

void Activation::initForward()
{	
	auto activation_param = _param.activation_param();
	cudnnActivationMode_t _activation_mode =  (cudnnActivationMode_t) activation_param.type();
	float coef = activation_param.coef();
	DF_CUDNN_CHECK(cudnnSetActivationDescriptor(_activation_desc, _activation_mode, CUDNN_PROPAGATE_NAN, coef));
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));

	std::string opString;
	switch (_activation_mode) {
	case CUDNN_ACTIVATION_SIGMOID:
		opString = "Sigmoid";
		break;
	case CUDNN_ACTIVATION_RELU:
		opString = "Relu";
		break;
	case CUDNN_ACTIVATION_TANH:
		opString = "Tanh";
		break;
	case CUDNN_ACTIVATION_CLIPPED_RELU:
		opString = "ClippedRelu";
		break;
	case CUDNN_ACTIVATION_ELU:
		opString = "Elu";
	};

	_outputs[0]->initValue(_inputs[0]->value()->dims());

	LOG(INFO) << "Initializing " << opString << " " << _name << " - " << _outputs[0]->value()->shape();
}

void Activation::initBackward()
{
	_outputs[0]->initDiff();
}

void Activation::forward()
{
	DF_CUDNN_CHECK(cudnnActivationForward(_cudnnHandle, _activation_desc, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}

void Activation::backward()
{
	DF_CUDNN_CHECK(cudnnActivationBackward(_cudnnHandle, _activation_desc, &alpha, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &beta, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));
}

std::string Activation::to_cpp() const
{
	auto activation_param = _param.activation_param();
	cudnnActivationMode_t _activation_mode = (cudnnActivationMode_t)activation_param.type();
	float coef = activation_param.coef();
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
	std::string cpp = "auto " + _name + " = df."+ op + "(" + _inputs[0]->connectedNode()->name() + ", ";
	if (op == "clipped_relu" || op == "elu")
		cpp += std::to_string(coef) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
