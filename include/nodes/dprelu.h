#pragma once

#include "core/node.h"

// Double Parametric Rectifier Unit
class DeepFlowDllExport DPRelu : public Node {
public:
	DPRelu(deepflow::NodeParam *param);
	int minNumInputs() override { return 3; }
	int minNumOutputs() override  { return 1; }
	std::string op_name() const override { return "dprelu"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
};