#include "core/node.h"
#include "core/common_cu.h"

__global__
void CpyAddKernel(const int n, const float alpha, const float *src, const float beta, float *dst)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = beta * dst[i] + alpha * src[i];
}

void Node::cpy(int n, const float alpha, const void * src, const float beta, void * dst)
{
	CpyAddKernel <<< numOfBlocks(n), maxThreadsPerBlock >>> (n, alpha, (float*) src, beta, (float*) dst);
	DF_KERNEL_CHECK();
}
