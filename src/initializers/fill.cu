#include "core/common_cu.h"

#include "initializers/fill.h"
#include "core/variable.h"

__global__
void FillKernel(const int n, float *a, const float v)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = v;
}

Fill::Fill(const InitParam &param) : Initializer(param) {
	LOG_IF(FATAL, param.has_fill_param() == false) << "param.has_fill_param() == false";		
}

void Fill::apply(Variable *variable) {
	float value = _param.fill_param().value();
	auto size = variable->output(0)->value()->size();
	LOG(INFO) << "Filling " << variable->name() << " with " << value;
	FillKernel <<< numOfBlocks(size), maxThreadsPerBlock >>> (size, (float*)variable->output(0)->value()->mutableData(), value);
	DF_KERNEL_CHECK();
}