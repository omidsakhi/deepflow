#pragma once

#include "core/node.h"
#include "proto/deepflow.pb.h"

class DeepFlowDllExport SoftmaxLoss : public Node {
public:
	SoftmaxLoss(NodeParam param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();	
	void initBackward();
	void forward();
	void backward();
private:
	cudnnHandle_t _cudnnHandle;	
	float alpha = 1.0f;
	float beta = 0.0f;
};