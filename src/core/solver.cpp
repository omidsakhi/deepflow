#include "core/solver.h"
#include "core/variable.h"

#include "observers/variable_finder.h"

#include <glog/logging.h>

#include <memory>

Solver::Solver(std::shared_ptr<OutputTerminal> loss, SolverParam param) {
	_param = param;
	_loss_terminal = loss;	
	_loss_node = loss->parentNode();
	_current_iteration = 0;

	VariableFinder _(&_variables);
	_loss_node->recursiveResetVisit();
	_loss_node->traverse(&_, TraverseOrder::PostOrder);
	LOG_IF(FATAL, _variables.size() == 0) << "Network has no variables.";
	LOG(INFO) << _variables.size() << " variables detected.";

}