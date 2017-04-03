#pragma once

#include "core/node_observer.h"

class DeepFlowDllExport BackwardObserver : public NodeObserver {
public:
	void apply(Node*node);
};