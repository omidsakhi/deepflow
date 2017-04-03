#pragma once

#include "core/node_observer.h"
#include "core/node.h"

class DeepFlowDllExport InitForwardObserver : public NodeObserver {
public:
	void apply(Node*node);
};