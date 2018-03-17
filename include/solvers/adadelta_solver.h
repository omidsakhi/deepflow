#pragma once


#include "core/solver.h"

class DeepFlowDllExport AdaDeltaSolver : public Solver {
public:
	AdaDeltaSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var, cudaStream_t stream = 0) override;
	void init(std::shared_ptr<Variable> var) override;
	std::string to_cpp() const override;
private:
	float * _h1 = NULL;
	float * _h2 = NULL;	
	deepflow::AdaDeltaSolverParam *_my_param;
};
