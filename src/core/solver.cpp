#include "core/solver.h"
#include "nodes/variable.h"

#include <glog/logging.h>

#include <memory>

Solver::Solver(const SolverParam &param) {
	_param = param;
}

const SolverParam& Solver::param() const {
	return _param;
}

const std::string Solver::name() const {
	return _param.name();
}

bool Solver::hasTheSameParam(std::shared_ptr<Solver> another) const
{
	return _param.name() == another->name();
}
