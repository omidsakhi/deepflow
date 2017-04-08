#include "core/common_cu.h"

#include "initializers/random_uniform.h"
#include "core/variable.h"

__global__ void RandKernel(const float min, const float max,const int n, float* dst) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(i, 0, 0, &state);
	if (i < n)
		dst[i] = (curand_uniform(&state) * (max - min)) + min;
}

RandomUniform::RandomUniform(const InitParam &param) : Initializer(param) {
	LOG_IF(FATAL, param.has_random_uniform_param() == false) << "param.has_random_uniform_param() == false";
}

void RandomUniform::apply(Variable *variable) {	
	auto size = variable->output(0)->value()->size();
	LOG(INFO) << "Random uniform filling " << variable->name() << " - " << variable->output(0)->value()->shape();
	float min = _param.random_uniform_param().min();
	float max = _param.random_uniform_param().max();
	RandKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (min, max, size, (float*)variable->output(0)->value()->mutableData());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0) << cudaGetErrorString(cudaPeekAtLastError());
}