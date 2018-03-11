#pragma once


#include "core/node.h"

class DeepFlowDllExport PassThrough : public Node {
public:
	PassThrough(deepflow::NodeParam *param);
	int minNumInputs() override { return 1; }
	int minNumOutputs() override { return 1; }
	std::string op_name() const override { return "pass_through"; }
	void init() override;	
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
private:
	bool _stop_gradients = true;
};
