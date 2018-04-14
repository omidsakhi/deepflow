#pragma once

#include "core/solver.h"

class DeepFlowDllExport SGDSolver : public Solver {
public:
	SGDSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var, cudaStream_t stream = 0) override;
	void init(std::shared_ptr<Variable> var) override;
	std::string to_cpp() const override;	
protected:
	deepflow::SGDSolverParam *_my_param;
	float * _h = NULL;
};