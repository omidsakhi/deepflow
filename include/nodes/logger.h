#pragma once


#include "core/node.h"

class DeepFlowDllExport Logger : public Node {
public:
	Logger(deepflow::NodeParam *param);
	int minNumInputs();
	int minNumOutputs() { return 0; }
	std::string op_name() const override { return "logger"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	std::string _filePath;
	int _num_inputs = 0;
	std::string _raw_message;
	deepflow::ActionTime _logging_time;
	deepflow::ActionType _logging_type;
};
