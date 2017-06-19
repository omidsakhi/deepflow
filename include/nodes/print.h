#pragma once


#include "core/node.h"

class DeepFlowDllExport Print : public Node {
public:
	Print(deepflow::NodeParam *param);
	int minNumInputs();
	int minNumOutputs() { return 0; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	int _num_inputs = 0;
	std::string _raw_message;
	deepflow::ActionTime _print_time;
	deepflow::ActionType _print_type;
};
