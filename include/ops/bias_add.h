#pragma once

#include "core/node.h"

class DeepFlowDllExport BiasAdd : public Node {
public:
	BiasAdd(NodeParam param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
};
