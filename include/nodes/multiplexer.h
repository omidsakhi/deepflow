#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Multiplexer : public Node {
public:
	Multiplexer(deepflow::NodeParam *param);
	int minNumInputs();
	int minNumOutputs() { return 1; }
	void init();	
	void forward();
	void backward();	
	std::list<std::shared_ptr<Node>> inputNodes() const;
	std::list<std::shared_ptr<Node>> outputNodes() const;
	std::string to_cpp() const;
	void selectInput(int input);
private:
	int _num_inputs = 0;
	size_t _output_size_in_bytes = -1;
	int _selected_input = -1;
};