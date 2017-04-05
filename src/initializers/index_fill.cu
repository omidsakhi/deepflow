#include "core/common_cu.h"

#include "initializers/index_fill.h"
#include "core/variable.h"

__global__
void IndexFillKernel(const int n, float *a,const float offset)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = offset + i;
}

IndexFill::IndexFill(const InitParam &param) : Initializer(param) {
	LOG_IF(FATAL, param.has_index_fill_param() == false);	
}

void IndexFill::apply(Variable *variable) {
	float offset = _param.index_fill_param().offset();
	auto size = variable->output(0)->value()->size();
	LOG(INFO) << "Filling variable " << variable->name() << " with " << offset;
	IndexFillKernel << <numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)variable->output(0)->value()->mutableData(), offset);
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);
}