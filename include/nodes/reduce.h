#pragma once

#include "core/node.h"

class DeepFlowDllExport Reduce : public Node {
public:
	Reduce(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }
	std::string op_name() const override;
	void init();	
	void forward();
	void backward();
	bool requiresIndices();
	std::string to_cpp() const;
private:
	cudnnHandle_t _cudnnHandle;
	cudnnReduceTensorOp_t _reduceTensorOp;
	cudnnReduceTensorDescriptor_t _reduceTensorDesciptor;
	cudnnReduceTensorIndices_t _reduceTensorIndices;
	deepflow::ReduceParam::OutputType _type;
	float *_d_workspace;
	size_t _workspaceSizeInBytes;
};