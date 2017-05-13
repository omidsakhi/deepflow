#pragma once

#include "core/node.h"

class DeepFlowDllExport Equal : public Node {
public:
	Equal(const deepflow::NodeParam &_block_param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
};
