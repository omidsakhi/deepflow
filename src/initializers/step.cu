#include "core/common_cu.h"

#include "initializers/step.h"
#include "core/variable.h"

__global__
void StepFillKernel(const int n, float *out, const float min, const float step)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) out[i] = min + i * step;
}

Step::Step(const InitParam &param) : Initializer(param) {
	LOG_IF(FATAL, param.has_step_param() == false);
}

void Step::init() {
	
}

void Step::apply(Variable *variable) {	
	int size = variable->output(0)->value()->size();
	float min = _param.step_param().min();
	float max = _param.step_param().max();
	LOG(INFO) << "Step fill variable " << variable->name() << " of shape " << variable->output(0)->value()->toString();
	StepFillKernel << <numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)variable->output(0)->value()->mutableData(), min, (max-min)/size);
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);
}