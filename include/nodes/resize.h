#pragma once

#include "core/node.h"

class DeepFlowDllExport Resize : public Node {
public:
	Resize(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
protected:
	float m_height_scale = 1;
	float m_width_scale = 1;
};