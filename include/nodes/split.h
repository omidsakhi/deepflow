#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Split : public Node {
public:
	Split(deepflow::NodeParam *param);
	int minNumInputs() override { return 1; }
	int minNumOutputs() override;
	std::string op_name() const override { return "split"; }
	void init() override;	
	void forward() override;
	void backward() override;
	std::string to_cpp() const;
private:	
	int _num_outputs = 0;
};