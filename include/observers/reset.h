#pragma once

#include "core/node_observer.h"
#include "core/node.h"

class DeepFlowDllExport ResetObserver : public NodeObserver {
public:
	void apply(std::shared_ptr<Node> node);
};