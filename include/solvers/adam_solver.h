#pragma once


#include "core/solver.h"
#include "core/reader.h"

class DeepFlowDllExport AdamSolver : public Solver {
public:
	AdamSolver(const SolverParam &param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
private:
	float * _m = NULL;
	float * _v = NULL;	
	AdamSolverParam _my_param;
};