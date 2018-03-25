#pragma once

#include "core/node.h"

class DeepFlowDllExport ConvGaussianKernel : public Node {
public:
	ConvGaussianKernel(deepflow::NodeParam *param);
	int minNumInputs() override { return 0; }
	int minNumOutputs() override { return 1; }
	void setSigma(float value);
	std::string op_name() const override { return "gaussian_weights"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const;
private:
	void generate();
	int _window_size = 3;
	float _current_sigma = 1;
	int _num_channels = 3;
};

