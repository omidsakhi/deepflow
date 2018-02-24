#pragma once

#include "core/node.h"

class DeepFlowDllExport ReplayMemory : public Node {
public:
	ReplayMemory(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "replay_memory"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	int _num_samples_per_batch = 0;
	int _capacity = 0;	
	size_t _input_head = 0;
	size_t _output_head = 0;
	size_t _mem_size;
	int _available_samples = 0;
	int _size_per_sample = 0;
	float *dev_memory;
};
