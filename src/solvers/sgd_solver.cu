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

SGDSolver::SGDSolver(std::shared_ptr<OutputTerminal> loss, const SolverParam &param) : Solver(loss, param) {
	LOG_IF(FATAL, param.has_sgd_solver() == false);
}

void SGDSolver::train_step() {
	if (_initialized == false) {
		init();
	}
	const SGDSolverParam &param = _param.sgd_solver();
	ResetObserver resetObserver;
	ForwardObserver forwardObserver;
	BackwardObserver backwardObserver;
	_loss_node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
	_loss_node->traverse(&forwardObserver, TraverseOrder::PostOrder, false);
	_loss_node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
	_loss_node->traverse(&backwardObserver, TraverseOrder::PreOrder, false);
	for(auto var : _variables) {
		auto output = var->output(0);
		auto size = output->value()->size();						
		ApplyGradientKernel << <(size + 255) / 256, 256 >> > (size, param.momentum(), param.learning_rate(), (float*) output->value()->mutableData(), (float*) output->diff()->data());
		LOG_IF(FATAL, cudaPeekAtLastError() != 0);		
		if (var->snapshot() && _current_iteration % var->snapshotInterval() == 0)
			var->toImage(_current_iteration);		
	}
	_current_iteration++;
}

void SGDSolver::init() {

}