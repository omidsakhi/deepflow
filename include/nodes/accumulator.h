#pragma once


#include "core/node.h"

class DeepFlowDllExport Accumulator : public Node {
public:
	Accumulator(const deepflow::NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	virtual ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	virtual BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	deepflow::ActionTime _reset_time;
	float _total = 0;
};