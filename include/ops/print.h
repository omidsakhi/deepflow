#pragma once


#include "core/node.h"

class DeepFlowDllExport Print : public Node {
public:
	Print(const NodeParam &param);	
	int minNumInputs();
	int minNumOutputs() { return 0; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
private:
	int _num_inputs = 0;
};
