#pragma once

#include "core/node.h"

class DeepFlowDllExport BiasAdd : public Node {
public:
	BiasAdd(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }	
	std::string op_name() const override { return "bias_add"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	int _inner_dim = 0;
	int _bias_dim = 0;	
	cudnnHandle_t _cudnnHandle;
};
