#include "solvers/adadelta_solver.h"
#include "core/common_cu.h"
#include "nodes/variable.h"

#include <glog/logging.h>

__global__
void FillKernel(int n, float *a)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = 1.0f;
}

__global__
void AdaDeltaKernel(const int n, float *w, const float *g, float *h1, float *h2, const float momentum, const float learning_rate, const float delta)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float gi = g[i];
		float hi = h1[i] = momentum * h1[i] + (1 - momentum) * gi * gi;
		gi = gi * sqrt((h2[i] + delta) / (hi + delta));
		h2[i] = momentum * h2[i] + (1 - momentum) * gi * gi;
		w[i] -= learning_rate * gi;
	}
}

AdaDeltaSolver::AdaDeltaSolver(deepflow::SolverParam *param) : Solver(param) {
	LOG_IF(FATAL, param->has_adadelta_solver() == false) << "param.has_adadelta_solver() == false";
	_my_param = param->mutable_adadelta_solver();
}

void AdaDeltaSolver::apply(std::shared_ptr<Variable> var) {
	auto context = var->executionContext();
	bool verbos = (context && context->debug_level > 3) ? true : false;
	if (_initialized == false) {
		LOG_IF(INFO, verbos) << "SOLVER " << name() << " FOR VARIABLE " << var->name();
		init(var);
	}
	if (_enable_input) {
		bool is_enable = _enable_input->value()->toFloat() >= 1;
		if (!is_enable) {
			LOG_IF(INFO, verbos) << "SOLVER " << name() << " **NOT** APPLIED ON " << var->name();
			return;
		}
	}
	LOG_IF(INFO, verbos) << "APPLYING SOLVER " << name() << " ON " << var->name();	
	auto size = var->output(0)->value()->size();
	AdaDeltaKernel << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)var->output(0)->value()->mutableData(), (float*)var->gradients(), _h1, _h2, _my_param->momentum(), _my_param->learning_rate(), _my_param->delta());	
	DF_KERNEL_CHECK();
	var->reset_gradients();
}

void AdaDeltaSolver::init(std::shared_ptr<Variable> var) {
	auto size = var->output(0)->value()->size();
	auto sizeInBytes = var->output(0)->value()->sizeInBytes();
	DF_CUDA_CHECK(cudaMalloc(&_h1, sizeInBytes));	
	FillKernel << <numOfBlocks(size), maxThreadsPerBlock >> >(size, _h1);
	DF_KERNEL_CHECK();
	DF_CUDA_CHECK(cudaMalloc(&_h2, sizeInBytes));	
	FillKernel << <numOfBlocks(size), maxThreadsPerBlock >> >(size, _h2);
	DF_KERNEL_CHECK();
	_initialized = true;
}

std::string AdaDeltaSolver::to_cpp() const
{	
	std::string cpp = "auto " + name() + " = df.adadelta_solver(";
	cpp += std::to_string(_my_param->learning_rate()) + ", ";
	cpp += std::to_string(_my_param->momentum()) + ", ";
	cpp += std::to_string(_my_param->delta()) + ", ";
	cpp += "\"" + name() + "\"";
	cpp += ");";
	return cpp;
}
