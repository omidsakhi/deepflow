#pragma once

#include "core/solver.h"
#include "core/reader.h"

class DeepFlowDllExport GainSolver : public Solver {
public:
	GainSolver(std::shared_ptr<OutputTerminal> loss, SolverParam param);
	void step();
	void train();
private:	
	std::list < std::shared_ptr<Reader>> _readers;
	std::vector<float *> _gains;
	std::vector<float *> _previous_gradients;
	std::vector<cudaStream_t> _streams;
	int maxThreadsPerBlock;
};