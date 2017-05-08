#pragma once

#include "core/node.h"

class DeepFlowDllExport Reduce : public Node {
public:
	Reduce(const deepflow::NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	bool requiresIndices();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cudnnHandle_t _cudnnHandle;
	cudnnReduceTensorOp_t _reduceTensorOp;
	cudnnReduceTensorDescriptor_t _reduceTensorDesciptor;
	cudnnReduceTensorIndices_t _reduceTensorIndices;
	deepflow::ReduceParam::OutputType _type;
	float *_d_workspace;
	size_t _workspaceSizeInBytes;
};