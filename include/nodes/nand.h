#pragma once

#include "core/node.h"

class DeepFlowDllExport Nand : public Node {
public:
	Nand(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "nand"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
};
