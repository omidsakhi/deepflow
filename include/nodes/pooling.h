#pragma once

#include "core/node.h"

class DeepFlowDllExport Pooling : public Node {
public:
	Pooling(const NodeParam &param);	
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	virtual BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
protected:
	cudnnHandle_t _cudnnHandle;	
	cudnnPoolingDescriptor_t _poolingDesc;
	const float _alpha = 1.0f;
	const float _beta = 0.0f;
};