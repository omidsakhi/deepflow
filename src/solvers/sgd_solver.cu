#include "solvers/sgd_solver.h"
#include "core/common_cu.h"

#include <algorithm>

#include "nodes/variable.h"

__global__
void ApplyGradientKernel(const int n, const float momentum, const float learning_rate, float *w, const float *grad)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) 
		w[i] = momentum * w[i] - learning_rate * grad[i];
}

SGDSolver::SGDSolver(deepflow::SolverParam *param) : Solver(param) {
	LOG_IF(FATAL, param->has_sgd_solver() == false) << "param.has_sgd_solver() == false";
	_my_param = _param->mutable_sgd_solver();
}

void SGDSolver::apply(std::shared_ptr<Variable> var) {
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
	ApplyGradientKernel << <numOfBlocks(size), maxThreadsPerBlock>> > (size, _my_param->momentum(), _my_param->learning_rate(), (float*)var->output(0)->value()->mutableData(), (float*)var->gradients());
	DF_KERNEL_CHECK();
	var->reset_gradients();
}

void SGDSolver::init(std::shared_ptr<Variable> var) {
	_initialized = true;
}

std::string SGDSolver::to_cpp() const
{
	std::string cpp = "auto " + name() + " = df.sgd_solver(";
	cpp += std::to_string(_my_param->momentum()) + ", ";
	cpp += std::to_string(_my_param->learning_rate()) + ", ";	
	cpp += "\"" + name() + "\"";
	cpp += ");";
	return cpp;
}
