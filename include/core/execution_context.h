#pragma once

#include "core/export.h"

#include <string>
#include <memory>

class DeepFlowDllExport ExecutionContext {
public:
	enum ExecutionPreference {
		PREFER_FASTEST = 0,
		PREFER_LIMITED_MEMORY = 1
	};
	enum ExecutionMode {
		TRAIN = 0,
		TEST = 1
	};
	int current_epoch = 1;	
	int current_iteration = 1;	
	int debug_level = 0;
	bool quit = false;	
	ExecutionPreference execution_preference = PREFER_FASTEST;
	ExecutionMode execution_mode = TRAIN;
};

using ExecutionContextPtr = std::shared_ptr<ExecutionContext>;