#pragma once

#include "core/node_observer.h"

class DeepFlowDllExport BackwardObserver : public NodeObserver {
public:
	void apply(std::shared_ptr<Node> node);
};