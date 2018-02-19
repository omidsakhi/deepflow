#pragma once


#include "core/node.h"

class DeepFlowDllExport Accumulator : public Node {
public:
	Accumulator(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }	
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	deepflow::ActionTime _reset_time;
	float _total = 0;
};