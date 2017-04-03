#pragma once

#include "core/export.h"
#include "core/initializer.h"

class DeepFlowDllExport IndexFill : public Initializer {
public:
	IndexFill(InitParam param);
	void apply(Variable *variable);
	void init() {}
};