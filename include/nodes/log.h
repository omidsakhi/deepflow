#pragma once


#include "core/node.h"

class DeepFlowDllExport Log : public Node {
public:
	Log(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "log"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
};

