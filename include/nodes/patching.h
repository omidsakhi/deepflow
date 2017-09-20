#pragma once

#include "core/node.h"

class DeepFlowDllExport Patching : public Node {
public:
	Patching(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	deepflow::PatchingParam_Mode _mode;
	int _num_vertical_patches;
	int _num_horizontal_patches;
};
