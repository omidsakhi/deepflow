#include "core/solver.h"
#include "core/variable.h"

#include "observers/node_finder.h"
#include "observers/reset.h"

#include <glog/logging.h>

#include <memory>

Solver::Solver(std::shared_ptr<OutputTerminal> loss,const SolverParam &param) {
	_param = param;
	_loss_terminal = loss;	
	_loss_node = loss->node();
	_current_iteration = 0;

	ResetObserver resetObserver;
	_loss_node->traverse(&resetObserver, TraverseOrder::PostOrder, true);

	NodeFinder<Variable> varObserver(_variables);		
	_loss_node->traverse(&varObserver, TraverseOrder::PostOrder, false);
	
	LOG_IF(FATAL, _variables.size() == 0) << "Network has no variables.";
	LOG(INFO) << _variables.size() << " variables detected.";

}

const SolverParam& Solver::param() const {
	return _param;
}