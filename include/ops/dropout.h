#pragma once

#include "core/node.h"

class DeepFlowDllExport Dropout : public Node {
public:
	Dropout(const NodeParam &param);	
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
private:
	cudnnHandle_t _cudnnHandle;	
	cudnnDropoutDescriptor_t _dropoutDesc;
	size_t _state_sizes_in_bytes;
	size_t _reserve_sizes_in_bytes;
	float alpha = 1.0f;
	float beta = 0.0f;
	float _dropout;
	float *d_states;
	float *d_reserve;
};