#pragma once

#include "core/loss.h"
#include "proto/deepflow.pb.h"

class DeepFlowDllExport SoftmaxLoss : public Loss {
public:
	SoftmaxLoss(const NodeParam &param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 2; }
	void initForward();	
	void initBackward();
	void forward();
	void backward();
private:
	cudnnHandle_t _cudnnHandle;	
	float alpha = 1.0f;
	float beta = 0.0f;
};