#pragma once

#include "core/node_observer.h"

class DeepFlowDllExport ForwardObserver : public NodeObserver {
public:
	void apply(Node*node);
};