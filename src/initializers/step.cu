#include "core/common_cu.h"

#include "initializers/step.h"
#include "nodes/variable.h"

__global__
void StepFillKernel(const int n, float *out, const float min, const float step)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) out[i] = min + i * step;
}

Step::Step(deepflow::InitParam *param) : Initializer(param) {
	LOG_IF(FATAL, param->has_step_param() == false) << "param.has_step_param() == false";
}

void Step::init() {
	
}

std::string Step::to_cpp() const
{
	std::string cpp = "df.step(";
	cpp += "{" + std::to_string(_dims[0]) + ", " + std::to_string(_dims[1]) + ", " + std::to_string(_dims[2]) + ", " + std::to_string(_dims[3]) + "}, ";
	float min = _param->step_param().min();
	float max = _param->step_param().max();
	LOG_IF(FATAL, max < min) << "max < min";
	cpp += std::to_string(min) + ", ";
	cpp += std::to_string(max);
	cpp += ")";
	return cpp;
}

void Step::apply(Node *node) {	
	int size = node->output(0)->value()->size();
	float min = _param->step_param().min();
	float max = _param->step_param().max();
	for (auto output : node->outputs()) {
		StepFillKernel << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)output->value()->gpu_data(), min, (max - min) / size);
		DF_KERNEL_CHECK();
	}
}