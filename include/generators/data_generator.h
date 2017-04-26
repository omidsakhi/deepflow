#pragma once

#include "core/generator.h"
#include "core/variable.h"

class Initializer;

class DeepFlowDllExport DataGenerator : public Generator, public Variable {
public:
	DataGenerator(std::shared_ptr<Initializer> initializer, const NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	void nextBatch();
	void initForward();
	void initBackward();	
	void forward();
	void backward();
	bool isLastBatch();
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType() { return ALWAYS_BACKWARD; }
private:
	int _current_batch = 0;
	int _num_total_samples = 0;
	int _num_batches = 0;
	int _batch_size = 0;
	bool _no_solver = false;
	bool _last_batch = false;
};
