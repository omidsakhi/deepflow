#pragma once

#include "core/node.h"

#include <random>

class DeepFlowDllExport PatchSampling : public Node {
public:
	PatchSampling(deepflow::NodeParam *param);
	int minNumInputs() override{ return 1; }
	int minNumOutputs() override { return 1; }
	std::string op_name() const override { return "patch_sampling"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
private:
	int _num_samples;	
	int _patch_height;
	int _patch_width;
	int _input_width;
	int _input_height;
	int _num_channels;
	int _patch_max_x;
	int _patch_max_y;
	int *_h_x_pos = nullptr;
	int *_h_y_pos = nullptr;
	int *_d_x_pos = nullptr;
	int *_d_y_pos = nullptr;
	std::random_device _random_device;
	std::mt19937 _generator;	
};