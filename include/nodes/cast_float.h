#pragma once

#include "core/node.h"

class DeepFlowDllExport CastFloat : public Node {
public:
	CastFloat(const NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	virtual BackwardType backwardType() { return NEVER_BACKWARD; }
};