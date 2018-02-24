#pragma once

#include "core/node.h"

class DeepFlowDllExport Reshape : public Node {
public:
	Reshape(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "reshape"; }
	void init();
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	std::array<int, 4> _output_dims;
};