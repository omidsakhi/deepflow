#pragma once

#include "core/node.h"

class DeepFlowDllExport BatchStdDev : public Node {
public:
	BatchStdDev(deepflow::NodeParam *param);
	int minNumInputs() override { return 1; }
	int minNumOutputs() override { return 1; }
	std::string op_name() const override { return "batch_stddev"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
private:
	cudnnHandle_t _cudnnHandle;	
	cudnnReduceTensorDescriptor_t _avgReduceTensorDesc1, _avgReduceTensorDesc2;
	cudnnTensorDescriptor_t _avgDesc1, _avgDesc2;
	float *_d_workspace = nullptr;
	float *_d_avg = nullptr;
	float *_d_std = nullptr;
	size_t _workspaceSizeInBytes = 0;
	int _inner_dim = 0;
};