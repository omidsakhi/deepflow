#pragma once

#include "core/node.h"

class DeepFlowDllExport Resize : public Node {
public:
	Resize(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "resize"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
protected:
	float m_height_scale = 1;
	float m_width_scale = 1;
};