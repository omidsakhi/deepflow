#pragma once

#include "core/node.h"

class DeepFlowDllExport Block : public Node {
public :
	Block(std::initializer_list<std::shared_ptr<InputTerminal>> inputs, std::initializer_list<std::shared_ptr<OutputTerminal>> outputs, NodeParam param);
	void initForward();
	void initBackward();
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 0; }
	void forward();
	void backward();
	void traverse(NodeObserver *observer, TraverseOrder order);
	void recursiveResetVisit();
};
