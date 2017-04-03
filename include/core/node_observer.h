#pragma once

#include "core/node.h"

#include <memory>

class DeepFlowDllExport NodeObserver {
public:
	virtual void apply(Node *node) = 0;
};