#pragma once

#include "core/node.h"

class DeepFlowDllExport EuclideanDistance : public Node {
public:
	EuclideanDistance(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return ALWAYS_BACKWARD; }
};