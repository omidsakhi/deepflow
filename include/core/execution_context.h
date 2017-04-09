#pragma once

#include "core/export.h"

#include <string>
#include <memory>

class DeepFlowDllExport ExecutionContext {
public:
	std::string phase;
	int current_epoch = 0;
	int current_iteration_per_epoch = 0;
	int current_iteration = 0;
	bool last_batch = false;
	int debug_level = 0;
};

using ExecutionContextPtr = std::shared_ptr<ExecutionContext>;