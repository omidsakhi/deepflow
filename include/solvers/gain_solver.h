#pragma once

#include "core/solver.h"
#include "core/reader.h"

class DeepFlowDllExport GainSolver : public Solver {
public:
	GainSolver(NodeOutputPtr loss, const SolverParam &param);
	void train_step();	
	void init();	
private:	
	std::vector<float *> _gains;
	std::vector<float *> _previous_gradients;	
	GainSolverParam _my_param;
};