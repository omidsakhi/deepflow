#pragma once

#include "core/node.h"

class DeepFlowDllExport CastFloat : public Node {
public:
	CastFloat(const deepflow::NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
};