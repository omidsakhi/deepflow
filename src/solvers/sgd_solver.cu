#include "solvers/sgd_solver.h"
#include "core/common_cu.h"

#include <algorithm>

#include "core/variable.h"

__global__
void ApplyGradientKernel(const int n, const float momentum, const float learning_rate, float *var, const float *grad)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) var[i] = momentum * var[i] + learning_rate * grad[i];
}

SGDSolver::SGDSolver(std::shared_ptr<OutputTerminal> loss, SolverParam param) : Solver(loss, param) {
	LOG_IF(FATAL, param.has_sgd_solver() == false);
}

void SGDSolver::step() {
	const SGDSolverParam &param = _param.sgd_solver();
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveForward();
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveBackward();
	std::list<Node*>::iterator i = _variables.begin();
	while (i != _variables.end())
	{
		Variable *var = dynamic_cast<Variable*>(*i);
		auto output = var->output(0);
		auto size = output->value()->size();						
		ApplyGradientKernel << <(size + 255) / 256, 256 >> > (size, param.momentum(), param.learning_rate(), (float*) output->value()->mutableData(), (float*) output->diff()->data());
		LOG_IF(FATAL, cudaPeekAtLastError() != 0);		
		if (var->snapshot() && _current_iteration % var->snapshotInterval() == 0)
			var->toImage(_current_iteration);
		i++;
	}
}

void SGDSolver::train() {
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveInitForward();
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveInitBackward();
	for (int i = 0; i < _param.max_iteration(); ++i)
	{
		_current_iteration = i;
		step();
	}
}