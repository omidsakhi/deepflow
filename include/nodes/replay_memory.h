#pragma once

#include "core/node.h"

class DeepFlowDllExport ReplayMemory : public Node {
public:
	ReplayMemory(const deepflow::NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	int _batch_size;
	int _capacity;	
	int _input_head = 0;
	int _output_head = 0;
	float *dev_memory;
};
