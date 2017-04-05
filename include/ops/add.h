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
protected:
	float _alpha, _beta;
};
