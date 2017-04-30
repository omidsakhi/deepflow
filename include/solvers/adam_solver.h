#pragma once

#include "core/solver.h"

class DeepFlowDllExport AdamSolver : public Solver {
public:
	AdamSolver(const SolverParam &param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
	std::string to_cpp() const;
private:
	float * _m = NULL;
	float * _v = NULL;	
	AdamSolverParam _my_param;
};