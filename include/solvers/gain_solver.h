#pragma once

#include "core/solver.h"
#include "core/reader.h"

class DeepFlowDllExport GainSolver : public Solver {
public:
	GainSolver(std::shared_ptr<OutputTerminal> loss, SolverParam param);
	void train_step();	
	void init();
private:	
	std::vector<float *> _gains;
	std::vector<float *> _previous_gradients;
	std::vector<cudaStream_t> _streams;	
};