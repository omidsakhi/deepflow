#pragma once

#include "core/solver.h"

class DeepFlowDllExport RMSPropSolver : public Solver {
public:
	RMSPropSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var, cudaStream_t stream = 0) override;
	void init(std::shared_ptr<Variable> var) override;
	std::string to_cpp() const override;
private:
	float * _h = nullptr;	
	deepflow::RMSPropSolverParam *_my_param;
};