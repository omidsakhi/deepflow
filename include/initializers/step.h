#pragma once

#include "core/export.h"
#include "core/initializer.h"

class DeepFlowDllExport Step : public Initializer {
public:
	Step(const InitParam &param);
	void apply(Variable *variable);
	void init();
};