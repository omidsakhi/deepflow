#pragma once

#include "core/node.h"

class DeepFlowDllExport Concate : public Node {
public:
	Concate(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "concate"; }
	void init();
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	int _first_input_channels;
	int _second_input_channels;
	int _output_channels;
	int _width, _height;
};
