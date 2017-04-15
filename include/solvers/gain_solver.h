#pragma once

#include "core/solver.h"
#include "core/reader.h"

class DeepFlowDllExport GainSolver : public Solver {
public:
	GainSolver(const SolverParam &param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
private:	
	float * _gain = NULL;
	float * _previous_gradient = NULL;	
	GainSolverParam _my_param;
};