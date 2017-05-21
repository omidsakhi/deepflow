#include "core/node.h"
#include "core/common_cu.h"

__global__
void CpyAddKernel(const int n, const float *src, float *dst)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] += src[i];
}

void Node::cpyAddDiff(int n, const void * src, void * dst)
{
	CpyAddKernel <<< numOfBlocks(n), maxThreadsPerBlock >>> (n, (float*) src, (float*) dst);
	DF_KERNEL_CHECK();
}
