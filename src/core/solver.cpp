#include "core/solver.h"
#include "core/variable.h"

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