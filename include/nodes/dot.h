#pragma once

#include "core/node.h"

class DeepFlowDllExport Dot : public Node {
public:
	Dot(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }	
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
};