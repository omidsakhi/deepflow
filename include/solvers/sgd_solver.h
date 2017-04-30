#pragma once

#include "core/solver.h"

class DeepFlowDllExport SGDSolver : public Solver {
public:
	SGDSolver(const SolverParam &param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
	std::string to_cpp() const;	
protected:
	SGDSolverParam _my_param;
};