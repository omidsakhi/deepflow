#pragma once

#include "core/node.h"

class DeepFlowDllExport GaborKernel : public Node {
public:
	GaborKernel(deepflow::NodeParam *param);
	int minNumInputs() override { return 0; }
	int minNumOutputs() override { return 1; }	
	std::string op_name() const override { return "gabor_kernel"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const;
private:
	void generate();
	int _window_size;	
	int _num_output_channels;
	int _num_orientations;
	int _num_scales;
	float *_d_orientations = nullptr;
	float *_d_scales = nullptr;
	float _sigma;
	float _phi;
	bool _apply_scale;
};

