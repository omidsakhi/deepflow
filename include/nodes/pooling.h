#pragma once

#include "core/node.h"

class DeepFlowDllExport Pooling : public Node {
public:
	Pooling(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	cudnnHandle_t _cudnnHandle;	
	cudnnPoolingDescriptor_t _poolingDesc;
};