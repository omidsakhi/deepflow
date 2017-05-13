#pragma once

#include "core/export.h"
#include "core/node.h"

class DeepFlowDllExport Phaseplexer : public Node {
public:
	Phaseplexer(const deepflow::NodeParam &_block_param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::list<std::shared_ptr<Node>> inputNodes() const;
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	std::map<std::string, NodeInputPtr> _map;
};
