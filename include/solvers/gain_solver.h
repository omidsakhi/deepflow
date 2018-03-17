#pragma once

#include "core/solver.h"

class DeepFlowDllExport GainSolver : public Solver {
public:
	GainSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var, cudaStream_t stream = 0) override;
	void init(std::shared_ptr<Variable> var) override;
	std::string to_cpp() const override;
private:	
	float * _gain = NULL;
	float * _previous_gradient = NULL;	
	deepflow::GainSolverParam *_my_param;
};