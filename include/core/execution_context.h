#pragma once

#include "core/export.h"

#include <string>
#include <memory>

class DeepFlowDllExport ExecutionContext {
public:
	std::string phase;
	deepflow::PhaseParam_PhaseBehaviour phase_behaviour = deepflow::PhaseParam_PhaseBehaviour_TRAIN;
	int current_epoch = 0;
	int current_iteration_per_epoch = 0;
	int current_iteration = 0;
	bool last_batch = false;
	int debug_level = 0;
	bool quit = false;
	bool stop_training = false;
};

using ExecutionContextPtr = std::shared_ptr<ExecutionContext>;