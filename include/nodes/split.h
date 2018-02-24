#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Split : public Node {
public:
	Split(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 2; }
	std::string op_name() const override { return "split"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	size_t m_size_in_bytes;
};