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
		w[i] -= gi;
	}
}

AdamSolver::AdamSolver(deepflow::SolverParam *param) : Solver(param) {
	LOG_IF(FATAL, param->has_adam_solver() == false) << "param.has_adam_solver() == false";
	_my_param = param->mutable_adam_solver();
}

void AdamSolver::apply(std::shared_ptr<Variable> var) {
	auto context = var->executionContext();
	bool verbos = (context && context->debug_level > 3) ? true : false;
	if (_initialized == false) {
		LOG_IF(INFO, verbos) << "SOLVER " << name() << " FOR VARIABLE " << var->name();
		init(var);
	}
	if (_enable_input) {
		float value = _enable_input->value()->toFloat();
		bool is_enable = value >= 1;
		if (!is_enable) {
			LOG_IF(INFO, verbos) << "SOLVER " << name() << " **NOT** APPLIED ON " << var->name();
			return;
		}
	}
	LOG_IF(INFO, verbos) << "APPLYING SOLVER " << name() << " ON " << var->name();
	auto size = var->output(0)->value()->size();	
	AdamKernel << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)var->output(0)->value()->mutableData(), (float*)var->gradients(), _m, _v, _my_param->beta1(), _my_param->beta2(), _my_param->eps(), _my_param->learning_rate());	
	DF_KERNEL_CHECK();	
	var->reset_gradients();
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

std::string AdamSolver::to_cpp() const
{	
	std::string cpp = "auto " + name() + " = df.adam_solver(";
	cpp += std::to_string(_my_param->learning_rate()) + ", ";
	cpp += std::to_string(_my_param->beta1()) + ", ";
	cpp += std::to_string(_my_param->beta2()) + ", ";
	cpp += std::to_string(_my_param->eps()) + ", ";
	cpp += "\"" + name() + "\"";
	cpp += ");";
	return cpp;
}
