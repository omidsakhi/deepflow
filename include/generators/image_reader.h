#pragma once


#include "core/generator.h"
#include "core/node.h"

class Initializer;

class DeepFlowDllExport ImageReader : public Generator, public Node {
public:
	ImageReader(const NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	void nextBatch();
	void initForward();
	void initBackward();
	void forward();
	void backward();
	bool isLastBatch();
private:	
	bool _last_batch = true;
};
