#include "core/common_cu.h"

#include "initializers/random_uniform.h"
#include "nodes/variable.h"

#include <random>

RandomUniform::RandomUniform(deepflow::InitParam *param) : Initializer(param) {
	LOG_IF(FATAL, param->has_random_uniform_param() == false) << "param.has_random_uniform_param() == false";
}

void RandomUniform::apply(Variable *variable) {	
	auto size = variable->output(0)->value()->size();
	float min = _param->random_uniform_param().min();
	float max = _param->random_uniform_param().max();
	LOG_IF(FATAL, max < min) << "max < min";
	std::mt19937 generator;
	generator.seed(std::random_device()());
	std::uniform_real_distribution<float> distribution(min, max);
	float *h_rand = new float[size];
	for (int i = 0; i < size; ++i)
		h_rand[i] = distribution(generator);
	DF_CUDA_CHECK(cudaMemcpy((float*)variable->output(0)->value()->mutableData(), h_rand, variable->output(0)->value()->sizeInBytes(), cudaMemcpyHostToDevice));
}

std::string RandomUniform::to_cpp() const
{
	std::string cpp = "df.random_uniform(";
	cpp += "{" + std::to_string(_dims[0]) + ", " + std::to_string(_dims[1]) + ", " + std::to_string(_dims[2]) + ", " + std::to_string(_dims[3]) + "}, ";
	float min = _param->random_uniform_param().min();
	float max = _param->random_uniform_param().max();
	cpp += std::to_string(min) + ", " + std::to_string(max) + ")";
	return cpp;
}
