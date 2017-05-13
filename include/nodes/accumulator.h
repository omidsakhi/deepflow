#pragma once


#include "core/node.h"

class DeepFlowDllExport Accumulator : public Node {
public:
	enum ResetTime {		
		EndOfEpoch = 0,
		Never = 1
	};
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
	ResetTime _reset_time;
	float _total = 0;
};