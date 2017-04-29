#include "solvers/adam_solver.h"

#include "core/common_cu.h"
#include "nodes/variable.h"

#include <glog/logging.h>

__global__
void AdamKernel(const int n, float *w, const float *g, float *m, float *v, const float beta1, const float beta2, const float eps, const float learning_rate)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float gi = g[i];
		float mi = m[i] = m[i] * beta1 + gi*(1 - beta1);
		float vi = v[i] = v[i] * beta2 + gi*gi*(1 - beta2);
		gi = learning_rate * mi / (sqrt(vi) + eps);
		w[i] = w[i] + gi;
	}
}

AdamSolver::AdamSolver(const SolverParam &param) : Solver(param) {
	LOG_IF(FATAL, param.has_adam_solver() == false) << "param.has_adam_solver() == false";
	_my_param = param.adam_solver();
}

void AdamSolver::apply(std::shared_ptr<Variable> var) {
	if (_initialized == false)
		init(var);
	auto output = var->output(0);
	auto size = output->value()->size();
	AdamKernel << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)output->value()->mutableData(), (float*)output->diff()->data(), _m, _v, _my_param.beta1(), _my_param.beta2(), _my_param.eps(), _my_param.learning_rate());
	DF_KERNEL_CHECK();	
}

void AdamSolver::init(std::shared_ptr<Variable> var) {
	auto size = var->output(0)->value()->size();
	auto sizeInBytes = var->output(0)->value()->sizeInBytes();
	DF_CUDA_CHECK(cudaMalloc(&_m, sizeInBytes));	
	DF_CUDA_CHECK(cudaMemset(_m, 0, sizeInBytes));
	DF_CUDA_CHECK(cudaMalloc(&_v, sizeInBytes));
	DF_CUDA_CHECK(cudaMemset(_v, 0, sizeInBytes));
	_initialized = true;
}