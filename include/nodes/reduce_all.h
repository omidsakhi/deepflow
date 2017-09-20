#pragma once

#include "core/node.h"

class DeepFlowDllExport ReduceAll : public Node {
public:
	ReduceAll(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	deepflow::ReduceAllParam_ReduceAllOp _reduce_op;
};

