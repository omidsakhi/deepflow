#pragma once

#include "core/node.h"

class DeepFlowDllExport SquareError : public Node {
public:
	SquareError(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "square_error"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
};