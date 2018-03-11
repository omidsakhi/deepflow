#pragma once

#include "core/node.h"

#include <random>

/*
	input 0: mean
	input 1: sigma
	
	output 0: gaussian
*/
class DeepFlowDllExport Gaussian : public Node {
public:
	Gaussian(deepflow::NodeParam *param);
	int minNumInputs() override { return 2; }
	int minNumOutputs() override { return 1; }
	std::string op_name() const override { return "gaussian"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
private:
	std::random_device _random_device;
	std::mt19937 _mt19937;
	float *_normal = nullptr;
	size_t _size;
};

