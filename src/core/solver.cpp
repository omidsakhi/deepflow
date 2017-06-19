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

bool Solver::hasTheSameParam(std::shared_ptr<Solver> another) const
{
	return _param->name() == another->name();
}

NodeInputPtr Solver::enable_input()
{
	return _enable_input;
}

void Solver::create_enable_input()
{
	_enable_input = std::make_shared<NodeInput>( nullptr, 0);
}
