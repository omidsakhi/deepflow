#include "solvers\rmsprop_solver.h"

#include "core/common_cu.h"
#include "nodes/variable.h"

#include <glog/logging.h>

__global__
void RMSPropKernel(const int n, float *w, const float *g, float *h, const float rms_decay, const float eps, const float learning_rate, const bool dry_run)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float gi = g[i];
		float hi = h[i] = rms_decay * h[i] * (1 - rms_decay) * gi * gi;		
		if (!dry_run)
			w[i] -= learning_rate * gi / (sqrt(hi) + eps);
	}
}


RMSPropSolver::RMSPropSolver(deepflow::SolverParam * param) : Solver(param) {
	LOG_IF(FATAL, param->has_rmsprop_solver() == false) << "param.has_rmsprop_solver() == false";
	_my_param = param->mutable_rmsprop_solver();
	_learning_rate = param->learning_rate();
}


void RMSPropSolver::apply(std::shared_ptr<Variable> var) {
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
	RMSPropKernel << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)var->output(0)->value()->mutableData(), (float*)var->gradients(), _h, _my_param->rms_decay(), _my_param->eps(), _learning_rate, dry_run);
	DF_KERNEL_CHECK();
	var->reset_gradients();
}

void RMSPropSolver::init(std::shared_ptr<Variable> var) {
	auto size = var->output(0)->value()->size();
	auto sizeInBytes = var->output(0)->value()->sizeInBytes();
	DF_CUDA_CHECK(cudaMalloc(&_h, sizeInBytes));
	DF_CUDA_CHECK(cudaMemset(_h, 0, sizeInBytes));
	_initialized = true;
}

std::string RMSPropSolver::to_cpp() const
{
	std::string cpp = "auto " + name() + " = df.rmsprop(";
	cpp += std::to_string(_param->learning_rate()) + ", ";
	cpp += std::to_string(_my_param->rms_decay()) + ", ";
	cpp += std::to_string(_my_param->eps()) + ", ";	
	cpp += "\"" + name() + "\"";
	cpp += ");";
	return cpp;
}
