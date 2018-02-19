#pragma once

#include "core/node.h"

class DeepFlowDllExport LRN : public Node {
public:
	LRN(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void init();
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	cudnnLRNDescriptor_t _normDesc;
	cudnnHandle_t _cudnnHandle;
};
