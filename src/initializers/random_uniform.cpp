#include "core/common_cu.h"

#include "initializers/random_uniform.h"
#include "core/variable.h"

RandomUniform::RandomUniform(const InitParam &param) : Initializer(param) {
	LOG_IF(FATAL, param.has_random_uniform_param() == false) << "param.has_random_uniform_param() == false";
}

void RandomUniform::apply(Variable *variable) {	
	auto size = variable->output(0)->value()->size();
	float min = _param.random_uniform_param().min();
	float max = _param.random_uniform_param().max();
	float *h_rand = new float[size];
	for (int i = 0; i < size; ++i)
		h_rand[i] = ((float)rand() / RAND_MAX) * (max - min) + min;
	DF_CUDA_CHECK(cudaMemcpy((float*)variable->output(0)->value()->mutableData(), h_rand, variable->output(0)->value()->sizeInBytes(), cudaMemcpyHostToDevice));
}