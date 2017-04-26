#pragma once

#include "core/node.h"

#include <cudnn.h>

class DeepFlowDllExport Softmax : public Node {
public:
	Softmax(const NodeParam &param);	
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }	
	virtual BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	cudnnHandle_t _cudnnHandle;	
	float _alpha;
	float _beta;
};