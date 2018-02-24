#pragma once

#include "core/node.h"

class DeepFlowDllExport Restructure : public Node {
public:
	Restructure(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "restructure"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	cudnnHandle_t _cudnnHandle;
	int _first_dim;
	int _second_dim;
};