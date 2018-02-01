#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Split : public Node {
public:
	Split(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	size_t m_size_in_bytes;
};