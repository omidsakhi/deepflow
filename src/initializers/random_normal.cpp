
#include "initializers/random_normal.h"
#include "nodes/variable.h"

RandomNormal::RandomNormal(deepflow::InitParam *param) : Initializer(param) {
	LOG_IF(FATAL, param->has_random_normal_param() == false) << "param.has_random_normal_param() == false";
}

void RandomNormal::apply(Node *node) {
	float mean = _param->random_normal_param().mean();
	float stddev = _param->random_normal_param().stddev();	
	generator.seed(rd());
	auto size = node->output(0)->value()->size();
	float *h_rand = new float[size];
	std::normal_distribution<float> distribution(mean, stddev);	
	for (auto output : node->outputs()) {		
		for (int i = 0; i < size; ++i)
			h_rand[i] = distribution(generator);
		DF_CUDA_CHECK(cudaMemcpy((float*)output->value()->gpu_data(), h_rand, output->value()->bytes(), cudaMemcpyHostToDevice));
	}
	delete [] h_rand;
}

std::string RandomNormal::to_cpp() const
{
	std::string cpp = "df.random_normal(";
	cpp += "{" + std::to_string(_dims[0]) + ", " + std::to_string(_dims[1]) + ", " + std::to_string(_dims[2]) + ", " + std::to_string(_dims[3]) + "}, ";
	float mean = _param->random_normal_param().mean();
	float stddev = _param->random_normal_param().stddev();
	cpp += std::to_string(mean) + ", " + std::to_string(stddev) + ")";
	return cpp;
}
