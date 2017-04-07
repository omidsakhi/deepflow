#pragma once

#include "core/solver.h"

class DeepFlowDllExport SGDSolver : public Solver {
public:
	SGDSolver(std::shared_ptr<NodeOutput> loss, const SolverParam &param);
	void train_step();
	void init();
};