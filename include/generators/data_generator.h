#pragma once

#include "nodes/variable.h"

class Initializer;

class DeepFlowDllExport DataGenerator : public Variable {
public:
	DataGenerator(std::shared_ptr<Initializer> initializer, const deepflow::NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	bool isGenerator();
	void initForward();
	void initBackward();	
	void forward();
	void backward();
	bool isLastBatch();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return ALWAYS_BACKWARD; }
private:
	int _current_batch = 0;
	int _num_total_samples = 0;
	int _num_batches = 0;
	int _batch_size = 0;
	bool _no_solver = false;
	bool _last_batch = false;
};
