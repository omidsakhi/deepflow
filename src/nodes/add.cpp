#include "core/common_cu.h"

#include "nodes/add.h"

Add::Add(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_add_param() == false) << "param.has_add_param() == false";	
}

std::string Add::op_name() const
{
	std::string op;
	if (_alpha == 1 && _beta == 1)
		op = "add";
	else if (_alpha == 1 && _beta == -1)
		op = "subtract";
	else {
		op = "add";		
	}
	return op;
}

void Add::init() {
	
	auto a = _inputs[0];
	auto ad = a->dims();

	auto b = _inputs[1];
	auto bd = b->dims();

	_alpha = _param->add_param().alpha();
	_beta = _param->add_param().beta();	

	LOG_IF(FATAL, a->value()->size() != b->value()->size()) << _name << " - Different input sizes: " << a->value()->shape() << " vs " << b->value()->shape() ;		

	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	cudnnCreateOpTensorDescriptor(&_opTensorDesc);
	cudnnSetOpTensorDescriptor(_opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
}

void Add::forward() {	
	DF_NODE_CUDNN_CHECK(
	cudnnOpTensor(_cudnnHandle, _opTensorDesc, &_alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(), &_beta, _inputs[1]->value()->descriptor(), _inputs[1]->value()->gpu_data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data())
	);
}

void Add::backward() {
	if (_inputs[0]->diff()) {
		cudaMemcpy(_inputs[0]->diff()->gpu_data(), _outputs[0]->diff()->gpu_data(), _outputs[0]->diff()->bytes(), cudaMemcpyDeviceToDevice);
		cudnnScaleTensor(_cudnnHandle, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->gpu_data(), &_alpha);
	}
	if (_inputs[1]->diff()) {
		cudaMemcpy(_inputs[1]->diff()->gpu_data(), _outputs[0]->diff()->gpu_data(), _outputs[0]->diff()->bytes(), cudaMemcpyDeviceToDevice);
		cudnnScaleTensor(_cudnnHandle, _inputs[1]->diff()->descriptor(), _inputs[1]->diff()->gpu_data(), &_beta);
	}
}

void Add::setAlpha(float alpha)
{
	_alpha = alpha;
}

void Add::setBeta(float beta)
{
	_beta = beta;
}

std::string Add::to_cpp() const
{
	std::string op;
	float print_alpha_beta = false;
	if (_alpha == 1 && _beta == 1)
		op = "add";
	else if (_alpha == 1 && _beta == -1)
		op = "subtract";
	else {
		op = "add";
		print_alpha_beta = true;
	}
	std::string cpp = "auto " + _name + " = df." + op + "(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	if (print_alpha_beta)
		cpp += std::to_string(_alpha) + ", " + std::to_string(_beta);
	cpp += "\"" + _name + "\");";	
	return cpp;
}
