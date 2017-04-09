#pragma once


#include "core/node.h"

class DeepFlowDllExport Accumulator : public Node {
public:
	enum ResetTime {		
		EndOfEpoch = 0,
		Never = 1
	};
	Accumulator(const NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
protected:
	ResetTime _reset_time;
};