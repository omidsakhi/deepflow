#pragma once

#include "core/loss.h"

class DeepFlowDllExport EuclideanLoss : public Loss {
public:
	EuclideanLoss(const NodeParam &param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
};