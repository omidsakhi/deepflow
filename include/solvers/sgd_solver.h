#pragma once

#include "core/solver.h"

class DeepFlowDllExport SGDSolver : public Solver {
public:
	SGDSolver(std::shared_ptr<OutputTerminal> loss, SolverParam param);
	void step();
	void train();
};