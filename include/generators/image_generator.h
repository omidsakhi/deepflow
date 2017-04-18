#pragma once

#include "core/generator.h"
#include "core/variable.h"

class Initializer;

class DeepFlowDllExport ImageGenerator : public Generator, public Variable {
public:
	ImageGenerator(std::shared_ptr<Initializer> initializer, const NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	void nextBatch();
	void initForward();
	void initBackward();	
	void forward();
	void backward();
	bool isLastBatch();
private:
	int _current_batch = 0;
	int _batch_size = 0;
	int _total_size = 0;
};
