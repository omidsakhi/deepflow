#include "initializers/constant.h"

#include "nodes/variable.h"

Constant::Constant(deepflow::InitParam * param) : Initializer(param)
{
	LOG_IF(FATAL, param->has_constant_param() == false) << "param->has_constant_fill_param() == false";
}

void Constant::apply(Node * node)
{
	LOG_IF(FATAL, _param->constant_param().values().size() != node->output(0)->value()->size());
	DF_CUDA_CHECK(
		cudaMemcpy(node->output(0)->value()->mutableData(), _param->constant_param().values().data(), node->output(0)->value()->sizeInBytes(), cudaMemcpyHostToDevice)
	);
}

std::string Constant::to_cpp() const
{
	return std::string();
}
