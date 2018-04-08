#pragma once


#include "core/node.h"

class DeepFlowDllExport Accumulator : public Node {
public:
	Accumulator(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }	
	std::string op_name() const override { return "accumulator"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
	void reset();
private:	
	float _total = 0;
};