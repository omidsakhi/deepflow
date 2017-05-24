#include "nodes/reshape.h"

#include "core/common_cu.h"

Reshape::Reshape(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_reshape_param() == false) << "param.reshape_param() == false";
}

void Reshape::initForward() {
	auto dims = _param.reshape_param().tensor_param();	
	_outputs[0]->initValue({ dims.dims(0), dims.dims(1), dims.dims(2), dims.dims(3) });
	LOG(INFO) << "Initializing Reshape " << _name << " - " << _inputs[0]->value()->shape() << " -> " << _outputs[0]->value()->shape();
}

void Reshape::initBackward() {
	_outputs[0]->initDiff();
}

void Reshape::forward() {	
	cpy(_inputs[0]->value()->size(), 1, _inputs[0]->value()->data(), 0, _outputs[0]->value()->mutableData());
	//DF_CUDNN_CHECK(cudnnTransformTensor(_cudnnHandle, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}

void Reshape::backward() {
	cpy(_outputs[0]->diff()->size(), 1, _outputs[0]->diff()->data(), 1, _inputs[0]->diff()->mutableData());
	//DF_CUDNN_CHECK(cudnnTransformTensor(_cudnnHandle, &one, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), &zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));
}

std::string Reshape::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.reshape(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
