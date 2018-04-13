#pragma once

#include "core/node.h"

#include <random>

class DeepFlowDllExport LeakyRelu : public Node {
public:
	LeakyRelu(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "leaky_relu"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	float _initial_negative_slope = 0;
	float _negative_slope = 0;	
	std::random_device _random_device;
};
