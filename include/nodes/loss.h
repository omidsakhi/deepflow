#pragma once

#include "core/node.h"

class DeepFlowDllExport Loss : public Node {
public:
	Loss(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "loss"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
	void set_alpha(float value);
	void set_beta(float value);
	float loss() const;
private:
	cudnnHandle_t _cudnnHandle;
	cudnnReduceTensorOp_t _reduceTensorOp;
	cudnnReduceTensorDescriptor_t _reduceTensorDesciptor;		
	float *_d_workspace;
	size_t _workspaceSizeInBytes;
	float _alpha = 1.0f;
	float _beta = 0.0f;
};
