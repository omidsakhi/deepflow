#pragma once

#include "core/node.h"

class DeepFlowDllExport Patching : public Node {
public:
	Patching(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "patching"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	deepflow::PatchingParam_Mode _mode;
	int _num_vertical_patches;
	int _num_horizontal_patches;
};
