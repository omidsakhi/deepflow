#pragma once

#include "core/solver.h"

class DeepFlowDllExport AdamSolver : public Solver {
public:
	AdamSolver(deepflow::SolverParam *param);
	void apply(std::shared_ptr<Variable> var) override;
	void init(std::shared_ptr<Variable> var) override;
	std::string to_cpp() const override;
private:
	float * _m = nullptr;
	float * _v = nullptr;	
	deepflow::AdamSolverParam *_my_param;	
};