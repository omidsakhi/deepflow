#pragma once

#include "core/solver.h"

class DeepFlowDllExport AdamSolver : public Solver {
public:
	AdamSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
	std::string to_cpp() const;
private:
	float * _m = NULL;
	float * _v = NULL;	
	deepflow::AdamSolverParam *_my_param;
};