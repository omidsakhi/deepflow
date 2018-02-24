#pragma once

#include "core/node.h"

class DeepFlowDllExport Abs : public Node {
public:
	Abs(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }	
	std::string op_name() const override { return "abs"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
};
