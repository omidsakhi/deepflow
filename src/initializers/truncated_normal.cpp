
#include "initializers/truncated_normal.h"
#include "nodes/variable.h"

TruncatedNormal::TruncatedNormal(deepflow::InitParam *param) : Initializer(param) {
	LOG_IF(FATAL, param->has_truncated_normal_param() == false) << "param.truncated_normal_param() == false";
}

void TruncatedNormal::apply(Variable *variable) {
	auto size = variable->output(0)->value()->size();
	float mean = _param->truncated_normal_param().mean();
	float stddev = _param->truncated_normal_param().stddev();
	generator.seed(rd());
	std::normal_distribution<float> distribution(mean, stddev);
	float *h_rand = new float[size];
	for (int i = 0; i < size; ++i) {
		float value = distribution(generator);
		while (value > stddev || value < -stddev)
			value = distribution(generator);
		h_rand[i] = value;
	}
	DF_CUDA_CHECK(cudaMemcpy((float*)variable->output(0)->value()->mutableData(), h_rand, variable->output(0)->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	delete[] h_rand;
}

std::string TruncatedNormal::to_cpp() const
{
	std::string cpp = "df.truncated_normal(";
	cpp += "{" + std::to_string(_dims[0]) + ", " + std::to_string(_dims[1]) + ", " + std::to_string(_dims[2]) + ", " + std::to_string(_dims[3]) + "}, ";
	float mean = _param->random_normal_param().mean();
	float stddev = _param->random_normal_param().stddev();
	cpp += std::to_string(mean) + ", " + std::to_string(stddev) + ")";
	return cpp;
}
