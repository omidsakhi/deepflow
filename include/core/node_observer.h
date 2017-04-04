#pragma once

#include "core/node.h"

#include <memory>

class DeepFlowDllExport NodeObserver {
public:
	virtual void apply(std::shared_ptr<Node> node) = 0;
};