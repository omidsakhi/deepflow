#pragma once

#include "core/node.h"

class DeepFlowDllExport Exp : public Node {
public:
	Exp(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }	
	std::string op_name() const override { return "exp"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
};
