#pragma once

#include "core/solver.h"

class DeepFlowDllExport SGDSolver : public Solver {
public:
	SGDSolver(const SolverParam &param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
protected:
	SGDSolverParam _my_param;
};