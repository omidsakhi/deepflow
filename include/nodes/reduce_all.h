#pragma once

#include "core/node.h"

class DeepFlowDllExport ReduceAll : public Node {
public:
	ReduceAll(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	deepflow::ReduceAllParam_ReduceAllOp _reduce_op;
};

