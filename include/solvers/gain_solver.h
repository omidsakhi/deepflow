#pragma once

#include "core/solver.h"

class DeepFlowDllExport GainSolver : public Solver {
public:
	GainSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
	std::string to_cpp() const;
private:	
	float * _gain = NULL;
	float * _previous_gradient = NULL;	
	deepflow::GainSolverParam *_my_param;
};