#pragma once


#include "core/solver.h"
#include "core/reader.h"

class DeepFlowDllExport AdaDeltaSolver : public Solver {
public:
	AdaDeltaSolver(const SolverParam &param);
	void apply(std::shared_ptr<Variable> var);
	void init(std::shared_ptr<Variable> var);
private:
	float * _h1 = NULL;
	float * _h2 = NULL;	
	AdaDeltaSolverParam _my_param;
};
