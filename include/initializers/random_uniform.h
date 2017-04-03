#pragma once

#include "core/export.h"
#include "core/initializer.h"
#include "proto/deepflow.pb.h"

class DeepFlowDllExport RandomUniform : public Initializer {
public:
	RandomUniform(InitParam param);
	void init() {}
	void apply(Variable *variable);
};