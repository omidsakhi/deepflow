#pragma once

#include "core/loss.h"

class DeepFlowDllExport EuclideanLoss : public Loss {
public:
	EuclideanLoss(const NodeParam &param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 0; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType() { return ALWAYS_BACKWARD; }
};