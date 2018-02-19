#pragma once


#include "core/node.h"

class DeepFlowDllExport Activation : public Node {
public:
	Activation(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }	
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	cudnnActivationDescriptor_t _activation_desc;
	cudnnHandle_t _cudnnHandle;
};
