#pragma once

#include "core/node.h"

class DeepFlowDllExport Block : public Node {
public :
	Block(std::initializer_list<NodeInputPtr> inputs, std::initializer_list<NodeOutputPtr> outputs, NodeParam param);
	void initForward();
	void initBackward();
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 0; }
	void forward();
	void backward();
	void traverse(NodeObserver *observer, TraverseOrder order, bool visit_condition);
	void recursiveResetVisit();
};
