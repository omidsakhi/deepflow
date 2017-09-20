#pragma once

#include "core/node.h"

class DeepFlowDllExport Loss : public Node {
public:
	Loss(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	cudnnHandle_t _cudnnHandle;
	cudnnReduceTensorOp_t _reduceTensorOp;
	cudnnReduceTensorDescriptor_t _reduceTensorDesciptor;
	cudnnReduceTensorIndices_t _reduceTensorIndices;
	deepflow::ReduceParam::OutputType _type;
	float *_d_workspace;
	size_t _workspaceSizeInBytes;
};
