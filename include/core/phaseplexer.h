#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Phaseplexer : public Node {
public:
	Phaseplexer(const NodeParam &param); 
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::list<std::shared_ptr<Node>> sourceNodes() const;
protected:
	std::map<std::string, NodeInputPtr> _map;
};
