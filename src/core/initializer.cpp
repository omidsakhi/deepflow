#include "core/initializer.h"

#include <glog/logging.h>

Initializer::Initializer(deepflow::InitParam *param) : CudaHelper() {
	_param = param;
	LOG_IF(FATAL, param->has_tensor_param() == false) << "param.has_tensor_param() == false [FAILED]";
	const deepflow::TensorParam &tensorParam = param->tensor_param();
	switch (tensorParam.dims_size()) {
	case 1:
		_dims = { 1,tensorParam.dims(0),1,1 };
		break;
	case 2:
		_dims = { tensorParam.dims(0),tensorParam.dims(1),1,1 };
		break;
	case 3:
		_dims = { tensorParam.dims(0), 1, tensorParam.dims(1),tensorParam.dims(2) };
		break;
	case 4:
		_dims = { tensorParam.dims(0),tensorParam.dims(1),tensorParam.dims(2),tensorParam.dims(3) };
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}	
}

std::array<int, 4> Initializer::dims() const {
	return _dims;
}

deepflow::InitParam * Initializer::param()
{
	return _param;
}
