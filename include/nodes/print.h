#pragma once


#include "core/node.h"

class DeepFlowDllExport Print : public Node {
public:
	enum PrintTime {
		EVERY_PASS = 0,
		END_OF_EPOCH = 1
	};
	enum PrintType {
		VALUES = 0,
		DIFFS = 1
	};
	Print(const deepflow::NodeParam &param);
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
	PrintTime _print_time;
	PrintType _print_type;
};
