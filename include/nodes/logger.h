#pragma once


#include "core/node.h"

class DeepFlowDllExport Logger : public Node {
public:
	enum LoggingTime {
		EVERY_PASS = 0,
		END_OF_EPOCH = 1
	};
	enum LoggingType {
		VALUES = 0,
		DIFFS = 1
	};
	Logger(const deepflow::NodeParam &param);
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
	std::string _filePath;
	int _num_inputs = 0;
	std::string _raw_message;
	LoggingTime _logging_time;
	LoggingType _logging_type;
};
