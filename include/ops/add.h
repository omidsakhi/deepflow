#pragma once

#include "core/node.h"

class DeepFlowDllExport Add : public Node {
public:
	Add(const NodeParam &param);	
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();	
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	virtual BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
protected:
	float _alpha, _beta;
};
