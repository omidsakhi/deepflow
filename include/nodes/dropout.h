#pragma once

#include "core/node.h"

class DeepFlowDllExport Dropout : public Node {
public:
	Dropout(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }	
	std::string op_name() const override { return "dropout"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	cudnnHandle_t _cudnnHandle;	
	cudnnDropoutDescriptor_t _dropoutDesc;
	size_t _state_sizes_in_bytes;
	size_t _reserve_sizes_in_bytes;
	float _dropout;	
	float *d_states;
	float *d_reserve;
};