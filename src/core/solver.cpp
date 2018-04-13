#include "core/solver.h"
#include "nodes/variable.h"

#include <glog/logging.h>

#include <memory>

Solver::Solver(deepflow::SolverParam *param) {
	_param = param;
}

deepflow::SolverParam* Solver::param() const {
	return _param;
}

const std::string Solver::name() const {
	return _param->name();
}

const std::string Solver::scope() const
{
	return _param->scope();	
}

bool Solver::has_the_same_param(std::shared_ptr<Solver> another) const
{
	return _param->name() == another->name();
}

void Solver::set_learning_rate(float lr)
{
	_learning_rate = lr;
}

void Solver::set_enabled(bool state)
{	
	_enabled = state;
}
