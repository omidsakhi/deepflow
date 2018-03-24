#include "solvers/adam_solver.h"

#include "core/common_cu.h"
#include "nodes/variable.h"

#include <glog/logging.h>

__global__
void AdamFillKernel(const int n, const float value, float *dst)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = value;
}

__global__
void AdamKernel(const int n, float *w, const float *g, float *m, float *v, const float beta1, const float beta2, const float eps, const float learning_rate, const bool dry_run)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float gi = g[i];
		float mi = m[i] = m[i] * beta1 + gi*(1 - beta1);
		float vi = v[i] = v[i] * beta2 + gi*gi*(1 - beta2);
		if (!dry_run)
			w[i] -= learning_rate * mi / (sqrt(vi) + eps);
	}
}

AdamSolver::AdamSolver(deepflow::SolverParam *param) : Solver(param) {
	LOG_IF(FATAL, param->has_adam_solver() == false) << "param.has_adam_solver() == false";
	_my_param = param->mutable_adam_solver();	
	_learning_rate = param->learning_rate();
}

void AdamSolver::apply(std::shared_ptr<Variable> var, cudaStream_t stream) {	
	auto context = var->executionContext();
	bool verbos = (context && context->debug_level > 3) ? true : false;
	bool dry_run = false;
	if (_initialized == false) {
		LOG_IF(INFO, verbos) << "solver " << name() << " for variable " << var->name();
		init(var);
		dry_run = true;
	}
	if (!_enabled)
		return;
	LOG_IF(INFO, verbos) << "applying solver " << name() << " on " << var->name();
	auto size = var->output(0)->value()->size();	
	AdamKernel << <numOfBlocks(size), maxThreadsPerBlock, 0, stream >> > (size, (float*)var->output(0)->value()->mutableData(), (float*)var->gradients(), _m, _v, _my_param->beta1(), _my_param->beta2(), _my_param->eps(), _learning_rate, dry_run);
	DF_KERNEL_CHECK();	
	var->reset_gradients(stream);
}

void AdamSolver::init(std::shared_ptr<Variable> var) {
	auto size = var->output(0)->value()->size();
	auto sizeInBytes = var->output(0)->value()->sizeInBytes();
	DF_CUDA_CHECK(cudaMalloc(&_m, sizeInBytes));	
	DF_CUDA_CHECK(cudaMemset(_m, 0, sizeInBytes));
	DF_CUDA_CHECK(cudaMalloc(&_v, sizeInBytes));
	AdamFillKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, 1, _v);
	DF_KERNEL_CHECK();	
	_initialized = true;
}

std::string AdamSolver::to_cpp() const
{	
	std::string cpp = "auto " + name() + " = df.adam_solver(";	
	cpp += std::to_string(_param->learning_rate()) + ", ";
	cpp += std::to_string(_my_param->beta1()) + ", ";
	cpp += std::to_string(_my_param->beta2()) + ", ";
	cpp += std::to_string(_my_param->eps()) + ", ";
	cpp += "\"" + name() + "\"";
	cpp += ");";
	return cpp;
}
