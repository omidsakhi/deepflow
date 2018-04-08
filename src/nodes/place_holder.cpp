#include "nodes/place_holder.h"

PlaceHolder::PlaceHolder(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_place_holder_param() == false) << "param.has_place_holder_param() == false";
}

void PlaceHolder::init() {
		
	std::array<int, 4> dims;
	auto placeHolderParam = _param->place_holder_param();
	auto tensorParam = placeHolderParam.tensor_param();
	switch (tensorParam.dims_size()) {
	case 1:
		dims = { 1,tensorParam.dims(0),1,1 };
		break;
	case 2:
		dims = { tensorParam.dims(0),tensorParam.dims(1),1,1 };
		break;
	case 3:
		dims = { tensorParam.dims(0), 1, tensorParam.dims(1),tensorParam.dims(2) };
		break;
	case 4:
		dims = { tensorParam.dims(0),tensorParam.dims(1),tensorParam.dims(2),tensorParam.dims(3) };
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}
	Tensor::TensorType type = (Tensor::TensorType) tensorParam.type();
	_outputs[0]->initValue(dims,type);
	_outputs[0]->initDiff();	

}

void PlaceHolder::forward() {
}

void PlaceHolder::backward() {
}

std::string PlaceHolder::to_cpp() const
{
	auto placeHolderParam = _param->place_holder_param();
	auto tensorParam = placeHolderParam.tensor_param();
	std::string dims;
	switch (tensorParam.dims_size()) {
	case 1:
		dims = "{1, " + std::to_string(tensorParam.dims(0)) + ", 1, 1}";
		break;
	case 2:
		dims = "{" + std::to_string(tensorParam.dims(0)) + ", " + std::to_string(tensorParam.dims(1)) + ", 1, 1}";
		break;
	case 3:
		dims = "{" + std::to_string(tensorParam.dims(0)) +", 1, " + std::to_string(tensorParam.dims(1)) + ", " + std::to_string(tensorParam.dims(2)) + "}";
		break;
	case 4:
		dims = "{" + std::to_string(tensorParam.dims(0)) +", " + std::to_string(tensorParam.dims(1)) + ", " + std::to_string(tensorParam.dims(2)) + ", "+ std::to_string(tensorParam.dims(3)) + "}";
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}

	std::string type;
	switch (tensorParam.type())
	{
	case deepflow::TensorParam_TensorType_FLOAT:
		type = "Tensor::Float";
		break;
	case deepflow::TensorParam_TensorType_INT32:
		type = "Tensor::Int32";
		break;
	default:
		LOG(FATAL) << "Unsupported type.";
	}
	std::string cpp = "auto " + _name + " = df.place_holder(" + dims + ", " + type + ", ";	
	cpp += "\"" + _name + "\");";	
	return cpp;
}
