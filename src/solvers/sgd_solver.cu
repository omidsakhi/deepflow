#include "solvers/sgd_solver.h"
#include "core/common_cu.h"

#include <algorithm>
#include "observers/forward.h"
#include "observers/backward.h"
#include "observers/reset.h"

#include "core/variable.h"

__global__
void ApplyGradientKernel(const int n, const float momentum, const float learning_rate, float *var, const float *grad)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) var[i] = momentum * var[i] + learning_rate * grad[i];
}

SGDSolver::SGDSolver(const SolverParam &param) : Solver(param) {
	LOG_IF(FATAL, param.has_sgd_solver() == false) << "param.has_sgd_solver() == false";
	_my_param = _param.sgd_solver();
}

void SGDSolver::apply(std::shared_ptr<Variable> var) {
	if (_initialized == false) {
		init(var);
	}	
	auto output = var->output(0);
	auto size = output->value()->size();						
	ApplyGradientKernel << <numOfBlocks(size), maxThreadsPerBlock>> > (size, _my_param.momentum(), _my_param.learning_rate(), (float*) output->value()->mutableData(), (float*) output->diff()->data());
	DF_KERNEL_CHECK();
}

void SGDSolver::init(std::shared_ptr<Variable> var) {
	_initialized = true;
}