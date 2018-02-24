#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Phaseplexer : public Node {
public:
	Phaseplexer(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "phaseplexer"; }
	void init();	
	void forward();
	void backward();
	std::list<std::shared_ptr<Node>> inputNodes() const;
	std::string to_cpp() const;
private:
	std::map<std::string, NodeInputPtr> _map;
};
